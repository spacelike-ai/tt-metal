from __future__ import annotations

import math

import torch
import torch.nn.functional as torch_functional
import torchvision.models
import ttnn
from loguru import logger


class TableTransformer(torch.nn.Module):
    def __init__(
        self,
        *,
        device: ttnn.Device,
        num_classes: int,
        num_queries: int,
    ) -> None:
        super().__init__()

        hidden_dim = 256

        self._device = device

        self.backbone = torchvision.models.resnet18()
        del self.backbone.fc

        self.input_proj = torch.nn.Conv2d(in_channels=512, out_channels=hidden_dim, kernel_size=1)

        self.transformer = torch.nn.Transformer(hidden_dim, 8, 6, 6)

        self.class_embed = torch.nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = TorchMultilayerPerceptron(
            in_features=hidden_dim,
            hidden_features=hidden_dim,
            out_features=4,
            num_layers=3,
        )

        self.query_embed = torch.nn.Embedding(num_queries, hidden_dim)

        self.register_load_state_dict_post_hook(lambda _module, _incompatible_keys: self._convert_state())

    def _convert_state(self) -> None:
        self.tt_input_proj = Linear.from_torch(
            weight=self.input_proj.weight.squeeze(3).squeeze(2),
            bias=self.input_proj.bias,
            dtype=ttnn.bfloat16,
            device=self._device,
        )
        self.tt_class_embed = Linear.from_torch_model(
            self.class_embed,
            dtype=ttnn.bfloat16,
            device=self._device,
        )
        self.tt_transformer = Transformer(self.transformer, device=self._device)
        self.tt_bbox_embed = MultilayerPerceptron.from_torch_model(
            self.bbox_embed,
            dtype=ttnn.bfloat16,
            device=self._device,
        )
        self.tt_query_embed = ttnn.from_torch(
            self.query_embed.weight.unsqueeze(0),
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
            device=self._device,
        )

    def __call__(self, inputs: ttnn.Tensor) -> ttnn.Tensor:
        batch_size = inputs.shape[0]

        logger.debug("running resnet...")

        # TODO: using torch resnet implementation for now
        torch_x = self.backbone.conv1(inputs)
        torch_x = self.backbone.bn1(torch_x)
        torch_x = self.backbone.relu(torch_x)
        torch_x = self.backbone.maxpool(torch_x)
        torch_x = self.backbone.layer1(torch_x)
        torch_x = self.backbone.layer2(torch_x)
        torch_x = self.backbone.layer3(torch_x)
        torch_x = self.backbone.layer4(torch_x)

        x = ttnn.from_torch(
            torch_x.permute(0, 2, 3, 1),
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
            device=self._device,
        )

        logger.debug("performing input projection...")
        proj = self.tt_input_proj(x)
        ttnn.deallocate(x)

        mask = torch.full(list(proj.shape)[:3], fill_value=False)
        pos = ttnn.from_torch(
            positional_encoding(mask),
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            device=self._device,
        )

        # TODO: using torch until it becomes clear how to best reshape a tilized tensor
        x = ttnn.from_torch(
            ttnn.to_torch(proj).flatten(start_dim=1, end_dim=2),
            layout=ttnn.TILE_LAYOUT,
            device=self._device,
        )
        ttnn.deallocate(proj)

        zero_decoder_input = ttnn.from_torch(
            torch.full(
                size=[batch_size, *list(self.tt_query_embed.shape)[1:]],
                fill_value=0.0,
            ),
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            device=self._device,
        )

        query_pos = ttnn.repeat(self.tt_query_embed, ttnn.Shape([batch_size, 1, 1]))

        logger.debug("running transformer...")
        transformer_output = self.tt_transformer(
            encoder_input=x,
            decoder_input=zero_decoder_input,
            pos=pos,
            query_pos=query_pos,
        )
        ttnn.deallocate(x)
        ttnn.deallocate(pos)
        ttnn.deallocate(query_pos)
        ttnn.deallocate(zero_decoder_input)

        logger.debug("running box detection...")
        logits = self.tt_class_embed(transformer_output)
        boxes = ttnn.sigmoid(self.tt_bbox_embed(transformer_output))

        transformer_output.deallocate()

        return {"logits": logits, "boxes": boxes}


class Attention:
    def __init__(self, attention: torch.nn.MultiheadAttention, /, *, device: ttnn.Device) -> None:
        self._num_heads = attention.num_heads
        self._head_dim = attention.head_dim

        self._query = Linear.from_torch(
            weight=attention.in_proj_weight[:256],
            bias=attention.in_proj_bias[:256],
            dtype=ttnn.bfloat16,
            device=device,
        )

        self._key = Linear.from_torch(
            weight=attention.in_proj_weight[256:512],
            bias=attention.in_proj_bias[256:512],
            dtype=ttnn.bfloat16,
            device=device,
        )

        self._value = Linear.from_torch(
            weight=attention.in_proj_weight[512:],
            bias=attention.in_proj_bias[512:],
            dtype=ttnn.bfloat16,
            device=device,
        )

        self._out = Linear.from_torch(
            weight=attention.out_proj.weight,
            bias=attention.out_proj.bias,
            dtype=ttnn.bfloat16,
            device=device,
        )

    def __call__(self, q: ttnn.Tensor, k: ttnn.Tensor, v: ttnn.Tensor) -> None:
        assert q.get_layout() == ttnn.TILE_LAYOUT
        assert k.get_layout() == ttnn.TILE_LAYOUT
        assert v.get_layout() == ttnn.TILE_LAYOUT

        q_sequence_length = q.shape[1]
        k_sequence_length = k.shape[1]

        q_proj = self._query(q)
        k_proj = self._key(k)
        v_proj = self._value(v)

        # Split_query_key_value_and_split_heads does not support distinct query, key,
        # and value sequence lengths, so we pad the query.
        # https://github.com/tenstorrent/tt-metal/issues/6115
        assert k_sequence_length >= q_sequence_length
        tile_padding = smallest_multiple(k_sequence_length, 32) - smallest_multiple(q_sequence_length, 32)
        unpadded_shape = list(q_proj.shape)
        unpadded_shape[1] = k_sequence_length
        padded_q_proj = ttnn.pad(q_proj, [(0, tile_padding), (0, 0)], 0)
        padded_q_proj = ttnn.reshape(
            padded_q_proj,
            ttnn.Shape(unpadded_shape, padded_q_proj.shape.with_tile_padding()),
        )
        ttnn.deallocate(q_proj)

        qkv_proj = ttnn.concat([padded_q_proj, k_proj, v_proj], dim=2)
        ttnn.deallocate(padded_q_proj)
        ttnn.deallocate(k_proj)
        ttnn.deallocate(v_proj)

        (q, k, v) = ttnn.transformer.split_query_key_value_and_split_heads(qkv_proj, num_heads=self._num_heads)
        ttnn.deallocate(qkv_proj)

        # revert padding of q
        padded_shape = list(q.shape.with_tile_padding())
        unpadded_shape = list(q.shape)
        unpadded_shape[2] = q_sequence_length
        q = ttnn.reshape(q, ttnn.Shape(unpadded_shape, padded_shape))

        attention_scores = ttnn.matmul(q, k)
        ttnn.deallocate(q)
        ttnn.deallocate(k)

        attention_probs = ttnn.transformer.attention_softmax(
            attention_scores, attention_mask=None, head_size=self._head_dim
        )
        ttnn.deallocate(attention_scores)

        pre_output = ttnn.matmul(attention_probs, v)
        ttnn.deallocate(attention_probs)
        ttnn.deallocate(v)

        concatenated_pre_output = ttnn.transformer.concatenate_heads(pre_output)
        ttnn.deallocate(pre_output)

        output = self._out(concatenated_pre_output)
        ttnn.deallocate(concatenated_pre_output)

        # remove padding which leads to erronous addition
        # TODO: get rid of this
        clean_output = ttnn.from_torch(
            ttnn.to_torch(output),
            layout=ttnn.TILE_LAYOUT,
            device=output.device(),
        )
        ttnn.deallocate(output)

        return clean_output


class Linear:
    @classmethod
    def from_torch_model(
        cls,
        model: torch.nn.Linear,
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.Device,
    ) -> Linear:
        return cls.from_torch(
            weight=model.weight,
            bias=model.bias,
            dtype=dtype,
            device=device,
        )

    @classmethod
    def from_torch(
        cls,
        *,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        dtype: ttnn.DataType | None = None,
        device: ttnn.Device,
    ) -> Linear:
        tt_weight = ttnn.from_torch(
            weight.transpose(0, 1),
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            device=device,
        )
        tt_bias = (
            None
            if bias is None
            else ttnn.from_torch(
                bias.unsqueeze(0),
                layout=ttnn.TILE_LAYOUT,
                dtype=dtype,
                device=device,
            )
        )

        return cls(weight=tt_weight, bias=tt_bias)

    def __init__(
        self,
        *,
        weight: ttnn.Tensor,
        bias: ttnn.Tensor | None = None,
    ) -> None:
        weight_shape = weight.shape

        self._in_features = weight_shape[0]
        self._out_features = weight_shape[1]

        self._weight = weight
        self._bias = bias

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        assert x.shape[-1] == self._in_features, "input tensor does not have the expected shape"

        return ttnn.linear(
            x,
            self._weight,
            bias=self._bias,
        )


class LayerNorm:
    @classmethod
    def from_torch_model(
        cls,
        model: torch.nn.Linear,
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.Device,
    ) -> LayerNorm:
        return cls.from_torch(
            weight=model.weight,
            bias=model.bias,
            epsilon=model.eps,
            dtype=dtype,
            device=device,
        )

    @classmethod
    def from_torch(
        cls,
        *,
        weight: torch.Tensor,
        bias: torch.Tensor,
        epsilon: float = 1e-05,
        dtype: ttnn.DataType | None = None,
        device: ttnn.Device,
    ) -> LayerNorm:
        return cls(
            weight=ttnn.from_torch(
                weight,
                layout=ttnn.TILE_LAYOUT,
                dtype=dtype,
                device=device,
            ),
            bias=ttnn.from_torch(
                bias,
                layout=ttnn.TILE_LAYOUT,
                dtype=dtype,
                device=device,
            ),
            epsilon=epsilon,
        )

    def __init__(
        self,
        *,
        weight: ttnn.Tensor,
        bias: ttnn.Tensor | None = None,
        epsilon: float = 1e-05,
    ) -> None:
        weight_shape = weight.shape

        self._in_features = weight_shape[0]
        self._out_features = weight_shape[1]

        self._weight = weight
        self._bias = bias
        self._epsilon = epsilon

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.layer_norm(
            input_tensor=x,
            weight=self._weight,
            bias=self._bias,
            epsilon=self._epsilon,
        )


class TransformerEncoderLayer:
    def __init__(self, layer: torch.nn.TransformerEncoderLayer, /, *, device: ttnn.Device) -> None:
        self._self_attention = Attention(layer.self_attn, device=device)
        self._norm1 = LayerNorm.from_torch_model(layer.norm1, device=device)
        self._norm2 = LayerNorm.from_torch_model(layer.norm2, device=device)
        self._linear1 = Linear.from_torch_model(layer.linear1, device=device)
        self._linear2 = Linear.from_torch_model(layer.linear2, device=device)

    def __call__(self, x: ttnn.Tensor, *, pos: ttnn.Tensor) -> ttnn.Tensor:
        nx = self._norm1(x)
        x += self._self_attention(nx + pos, nx + pos, nx)
        ttnn.deallocate(nx)

        nx = self._norm2(x)
        x += self._linear2(ttnn.relu(self._linear1(nx)))
        ttnn.deallocate(nx)

        return x


class TransformerDecoderLayer:
    def __init__(self, layer: torch.nn.TransformerDecoderLayer, /, *, device: ttnn.Device) -> None:
        self._device = device
        self._self_attention = Attention(layer.self_attn, device=device)
        self._cross_attention = Attention(layer.multihead_attn, device=device)
        self._norm1 = LayerNorm.from_torch_model(layer.norm1, device=device)
        self._norm2 = LayerNorm.from_torch_model(layer.norm2, device=device)
        self._norm3 = LayerNorm.from_torch_model(layer.norm3, device=device)
        self._linear1 = Linear.from_torch_model(layer.linear1, device=device)
        self._linear2 = Linear.from_torch_model(layer.linear2, device=device)

    def __call__(
        self,
        x: ttnn.Tensor,
        *,
        mem: ttnn.Tensor,
        pos: ttnn.Tensor,
        query_pos: ttnn.Tensor,
    ) -> ttnn.Tensor:
        # ttnn.Tensor.__iadd__ currently does not modify the tensor in-place but
        # allocates a new one instead, which seems inefficient.

        nx = self._norm1(x)
        x += self._self_attention(nx + query_pos, nx + query_pos, nx)

        nx = self._norm2(x)
        x += self._cross_attention(nx + query_pos, mem + pos, mem)

        nx = self._norm3(x)
        x += self._linear2(ttnn.relu(self._linear1(nx)))

        return x


class Transformer:
    def __init__(self, transformer: torch.nn.Transformer, /, *, device: ttnn.Device) -> None:
        self._encoder_norm = LayerNorm.from_torch_model(transformer.encoder.norm, device=device)
        self._decoder_norm = LayerNorm.from_torch_model(transformer.decoder.norm, device=device)

        self._encoder_layers = [TransformerEncoderLayer(layer, device=device) for layer in transformer.encoder.layers]
        self._decoder_layers = [TransformerDecoderLayer(layer, device=device) for layer in transformer.decoder.layers]

    def __call__(
        self,
        *,
        encoder_input: ttnn.Tensor,
        decoder_input: ttnn.Tensor,
        pos: ttnn.Tensor,
        query_pos: ttnn.Tensor,
    ) -> ttnn.Tensor:
        x = encoder_input
        for layer in self._encoder_layers:
            x = layer(x, pos=pos)

        mem = self._encoder_norm(x)

        x = decoder_input
        for layer in self._decoder_layers:
            x = layer(x, mem=mem, pos=pos, query_pos=query_pos)

        return self._decoder_norm(x)


class MultilayerPerceptron:
    """Multi-layer perceptron with ReLU activation."""

    @classmethod
    def from_torch_model(
        cls,
        mlp: TorchMultilayerPerceptron,
        *,
        dtype: ttnn.DataType | None = None,
        device: ttnn.Device,
    ) -> MultilayerPerceptron:
        layers = [Linear.from_torch_model(layer, dtype=dtype, device=device) for layer in mlp.layers]

        return cls(layers=layers)

    def __init__(
        self,
        *,
        layers: list[Linear],
    ) -> None:
        self._layers = layers

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        for i, layer in enumerate(self._layers):
            x = layer(x)
            if i < len(self._layers) - 1:
                x = ttnn.relu(x)

        return x


class TorchMultilayerPerceptron(torch.nn.Module):
    """Multi-layer perceptron with ReLU activation."""

    def __init__(
        self,
        *,
        in_features: int,
        hidden_features: int,
        out_features: int,
        num_layers: int,
    ) -> None:
        super().__init__()

        features_list = [hidden_features] * (num_layers - 1)

        self.layers = torch.nn.ModuleList(
            torch.nn.Linear(i, o)
            for i, o in zip(
                [in_features, *features_list],
                [*features_list, out_features],
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch_functional.relu(x)

        return x


def smallest_multiple(x: int, factor: int) -> int:
    """Return smallest multiple of `factor` bigger or equal to `x`."""
    return (x + factor - 1) // factor * factor


def positional_encoding(mask: torch.Tensor) -> torch.Tensor:
    num_pos_feats = 128
    temperature = 10000

    not_mask = ~mask

    y_embed = not_mask.cumsum(1, dtype=torch.bfloat16)
    x_embed = not_mask.cumsum(2, dtype=torch.bfloat16)

    eps = 1e-6
    y_embed = y_embed / (y_embed[:, -1:, :] + eps) * 2 * math.pi
    x_embed = x_embed / (x_embed[:, :, -1:] + eps) * 2 * math.pi

    dim_t = torch.arange(num_pos_feats, dtype=torch.bfloat16)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

    pos_x = x_embed[:, :, :, None] / dim_t
    pos_y = y_embed[:, :, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(start_dim=3)
    pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(start_dim=3)

    return torch.cat((pos_y, pos_x), dim=3).flatten(start_dim=1, end_dim=2)


def unsqueeze(t: ttnn.Tensor) -> ttnn.Tensor:
    unpadded_shape = list(t.shape)
    padded_shape = list(t.shape.with_tile_padding())

    return ttnn.reshape(
        t,
        ttnn.Shape(
            [1, *list(unpadded_shape)],
            [1, *list(padded_shape)],
        ),
    )
