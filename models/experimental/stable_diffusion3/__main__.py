from __future__ import annotations

import hashlib
import json
import logging
import sys

import numpy as np
import torch
import tqdm
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.pipelines.stable_diffusion_3.pipeline_output import (
    StableDiffusion3PipelineOutput,
)
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
    StableDiffusion3Pipeline,
)
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from PIL import Image
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)

from .reference import SD3Transformer2DModel

CHECKPOINT = "stabilityai/stable-diffusion-3.5-medium"

logger = logging.getLogger(__name__)


# from datasets import load_dataset
# from PIL import Image
# from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

# import ttnn
# from models.demos.segformer.reference.segformer_for_semantic_segmentation import (
#     SegformerForSemanticSegmentationReference,
# )
# from models.demos.segformer.tt.ttnn_segformer_for_semantic_segmentation import (
#     TtSegformerForSemanticSegmentation,
# )
# from models.utility_functions import skip_for_grayskull
# from tests.ttnn.integration_tests.segformer.test_segformer_decode_head import (
#     create_custom_preprocessor as create_custom_preprocessor_deocde_head,
# )
# from tests.ttnn.integration_tests.segformer.test_segformer_model import (
#     create_custom_preprocessor as create_custom_preprocessor_model,
# )
# from tests.ttnn.utils_for_testing import assert_with_pcc
# from ttnn.model_preprocessing import ParameterDict, ParameterList, preprocess_model_parameters


# def create_custom_preprocessor(device):
#     def custom_preprocessor(model, name, ttnn_module_args):
#         parameters = {}
#         if isinstance(model, SegformerForSemanticSegmentationReference):
#             parameters["segformer"] = {}
#             segformer_preprocess = create_custom_preprocessor_model(device)
#             parameters["segformer"] = segformer_preprocess(model.segformer, None, None)
#             parameters["decode_head"] = {}
#             deocde_preprocess = create_custom_preprocessor_deocde_head(device)
#             parameters["decode_head"] = deocde_preprocess(model.decode_head, None, None)

#         return parameters

#     return custom_preprocessor


# def move_to_device(object, device):
#     if isinstance(object, ParameterDict):
#         for name, value in list(object.items()):
#             if name in ["sr", "proj", "dwconv", "linear_fuse", "classifier"]:
#                 continue
#             object[name] = move_to_device(value, device)
#         return object
#     elif isinstance(object, ParameterList):
#         for index, element in enumerate(object):
#             object[index] = move_to_device(element, device)
#         return object
#     elif isinstance(object, ttnn.Tensor):
#         return ttnn.to_device(object, device)
#     else:
#         return object


# @pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
# def run(device):
def run(*, device: str | torch.device, local_files_only: bool = True) -> None:  # noqa: PLR0915
    dtype = torch.bfloat16

    model_input = {
        "prompt_1": ["cat"],
        "prompt_2": ["cat"],
        "prompt_3": ["cat"],
        "negative_prompt_1": ["mouse"],
        "negative_prompt_2": ["mouse"],
        "negative_prompt_3": ["mouse"],
        "width": 512,  # default = 1024, works with 512
        "height": 512,  # default = 1024
        "num_inference_steps": 5,  # default = 50
        "guidance_scale": 7.0,
        "num_images_per_prompt": 1,
        "max_t5_sequence_length": 256,
        "seed": 1,
    }
    input_hash = hashlib.sha256(json.dumps(model_input).encode()).hexdigest()[:16]
    reference_image_path = f"{input_hash}-{dtype}-reference.png"

    try:
        reference_image = Image.open(reference_image_path)
    except FileNotFoundError:
        logger.info("generating reference image...")
        pipeline = StableDiffusion3Pipeline.from_pretrained(
            CHECKPOINT, torch_dtype=dtype, local_files_only=local_files_only
        ).to(device=device)
        assert isinstance(pipeline, StableDiffusion3Pipeline)
        torch.manual_seed(model_input["seed"])
        hf_output = pipeline(
            prompt=model_input["prompt_1"],
            prompt_2=model_input["prompt_2"],
            prompt_3=model_input["prompt_3"],
            negative_prompt=model_input["negative_prompt_1"],
            negative_prompt_2=model_input["negative_prompt_2"],
            negative_prompt_3=model_input["negative_prompt_3"],
            width=model_input["width"],
            height=model_input["height"],
            num_inference_steps=model_input["num_inference_steps"],
            guidance_scale=model_input["guidance_scale"],
            num_images_per_prompt=model_input["num_images_per_prompt"],
            max_sequence_length=model_input["max_t5_sequence_length"],
        )
        assert isinstance(hf_output, StableDiffusion3PipelineOutput)
        reference_image = hf_output.images[0]
        reference_image.save(reference_image_path)

    logger.info("loading models")
    tokenizer_1 = CLIPTokenizer.from_pretrained(
        CHECKPOINT,
        subfolder="tokenizer",
        local_files_only=local_files_only,
    )
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        CHECKPOINT,
        subfolder="tokenizer_2",
        local_files_only=local_files_only,
    )
    tokenizer_3 = T5TokenizerFast.from_pretrained(
        CHECKPOINT,
        subfolder="tokenizer_3",
        local_files_only=local_files_only,
    )
    text_encoder_1 = CLIPTextModelWithProjection.from_pretrained(
        CHECKPOINT,
        subfolder="text_encoder",
        torch_dtype=dtype,
        local_files_only=local_files_only,
    )
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        CHECKPOINT,
        subfolder="text_encoder_2",
        torch_dtype=dtype,
        local_files_only=local_files_only,
    )
    text_encoder_3 = T5EncoderModel.from_pretrained(
        CHECKPOINT,
        subfolder="text_encoder_3",
        torch_dtype=dtype,
        local_files_only=local_files_only,
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        CHECKPOINT,
        subfolder="scheduler",
        torch_dtype=dtype,
        local_files_only=local_files_only,
    )
    vae = AutoencoderKL.from_pretrained(
        CHECKPOINT,
        subfolder="vae",
        torch_dtype=dtype,
        local_files_only=local_files_only,
    )
    logger.info("loading transformer")
    transformer = SD3Transformer2DModel.from_pretrained(
        CHECKPOINT,
        subfolder="transformer",
        torch_dtype=dtype,
        local_files_only=local_files_only,
    )
    logger.info("loading done")

    assert isinstance(transformer, SD3Transformer2DModel)
    assert isinstance(tokenizer_1, CLIPTokenizer)
    assert isinstance(tokenizer_2, CLIPTokenizer)
    assert isinstance(tokenizer_3, T5TokenizerFast)
    assert isinstance(text_encoder_1, CLIPTextModelWithProjection)
    assert isinstance(text_encoder_2, CLIPTextModelWithProjection)
    assert isinstance(text_encoder_3, T5EncoderModel)
    assert isinstance(scheduler, FlowMatchEulerDiscreteScheduler)
    assert isinstance(vae, AutoencoderKL)

    text_encoder_1.to(device=device)  # type: ignore  # noqa: PGH003
    text_encoder_2.to(device=device)  # type: ignore  # noqa: PGH003
    text_encoder_3.to(device=device)  # type: ignore  # noqa: PGH003
    vae.to(device=device)  # type: ignore  # noqa: PGH003
    transformer.to(device=device)  # type: ignore  # noqa: PGH003

    block_out_channels = vae.config.block_out_channels  # type: ignore  # noqa: PGH003
    vae_scaling_factor = vae.config.scaling_factor  # type: ignore  # noqa: PGH003
    vae_shift_factor = vae.config.shift_factor  # type: ignore  # noqa: PGH003

    vae_scale_factor = 2 ** (len(block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

    assert model_input["height"] % (vae_scale_factor * transformer.patch_size) == 0
    assert model_input["width"] % (vae_scale_factor * transformer.patch_size) == 0
    assert model_input["max_t5_sequence_length"] <= 512  # noqa: PLR2004

    batch_size = len(model_input["prompt_1"])
    do_classifier_free_guidance = model_input["guidance_scale"] > 1
    tokenizer_max_length = tokenizer_1.model_max_length

    logger.info("encode prompts")

    logger.info("prompt 1")
    prompt_embed, pooled_prompt_embed = _get_clip_prompt_embeds(
        prompt=model_input["prompt_1"],
        device=device,
        num_images_per_prompt=model_input["num_images_per_prompt"],
        tokenizer=tokenizer_1,
        text_encoder=text_encoder_1,
        tokenizer_max_length=tokenizer_max_length,
    )

    logger.info("prompt 2")
    prompt_2_embed, pooled_prompt_2_embed = _get_clip_prompt_embeds(
        prompt=model_input["prompt_2"],
        device=device,
        num_images_per_prompt=model_input["num_images_per_prompt"],
        tokenizer=tokenizer_2,
        text_encoder=text_encoder_2,
        tokenizer_max_length=tokenizer_max_length,
    )
    clip_prompt_embeds = torch.cat([prompt_embed, prompt_2_embed], dim=-1)

    logger.info("prompt 3")
    t5_prompt_embed = _get_t5_prompt_embeds(
        prompt=model_input["prompt_3"],
        num_images_per_prompt=model_input["num_images_per_prompt"],
        max_sequence_length=model_input["max_t5_sequence_length"],
        device=device,
        tokenizer=tokenizer_3,
        text_encoder=text_encoder_3,
    )

    clip_prompt_embeds = torch.nn.functional.pad(
        clip_prompt_embeds,
        (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1]),
    )

    prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
    pooled_prompt_embeds = torch.cat([pooled_prompt_embed, pooled_prompt_2_embed], dim=-1)

    if do_classifier_free_guidance:
        logger.info("negative prompt 1")
        negative_prompt_embed, negative_pooled_prompt_embed = _get_clip_prompt_embeds(
            prompt=model_input["negative_prompt_1"],
            device=device,
            num_images_per_prompt=model_input["num_images_per_prompt"],
            tokenizer=tokenizer_1,
            text_encoder=text_encoder_1,
            tokenizer_max_length=tokenizer_max_length,
        )
        logger.info("negative prompt 2")
        negative_prompt_2_embed, negative_pooled_prompt_2_embed = _get_clip_prompt_embeds(
            prompt=model_input["negative_prompt_2"],
            device=device,
            num_images_per_prompt=model_input["num_images_per_prompt"],
            tokenizer=tokenizer_2,
            text_encoder=text_encoder_2,
            tokenizer_max_length=tokenizer_max_length,
        )
        negative_clip_prompt_embeds = torch.cat([negative_prompt_embed, negative_prompt_2_embed], dim=-1)

        logger.info("negative prompt 3")
        t5_negative_prompt_embed = _get_t5_prompt_embeds(
            prompt=model_input["negative_prompt_3"],
            num_images_per_prompt=model_input["num_images_per_prompt"],
            max_sequence_length=model_input["max_t5_sequence_length"],
            device=device,
            tokenizer=tokenizer_3,
            text_encoder=text_encoder_3,
        )

        negative_clip_prompt_embeds = torch.nn.functional.pad(
            negative_clip_prompt_embeds,
            (
                0,
                t5_negative_prompt_embed.shape[-1] - negative_clip_prompt_embeds.shape[-1],
            ),
        )

        negative_prompt_embeds = torch.cat([negative_clip_prompt_embeds, t5_negative_prompt_embed], dim=-2)
        negative_pooled_prompt_embeds = torch.cat(
            [negative_pooled_prompt_embed, negative_pooled_prompt_2_embed], dim=-1
        )

        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

    logger.info("prepare timesteps")

    scheduler.set_timesteps(model_input["num_inference_steps"], device=device)
    timesteps = scheduler.timesteps

    logger.info("prepare latents")

    num_channels_latents = transformer.in_channels

    latents_shape = (
        batch_size * model_input["num_images_per_prompt"],
        num_channels_latents,
        int(model_input["height"]) // vae_scale_factor,
        int(model_input["width"]) // vae_scale_factor,
    )
    torch.manual_seed(model_input["seed"])
    latents = torch.randn(latents_shape, dtype=prompt_embeds.dtype, device=device)

    logger.info("denoising loop")

    for t in tqdm.tqdm(timesteps):
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

        timestep = t.expand(latent_model_input.shape[0])

        noise_pred = transformer.forward(
            hidden_states=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
        )

        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + model_input["guidance_scale"] * (noise_pred_text - noise_pred_uncond)

        latents = scheduler.step(
            noise_pred,  # type: ignore  # noqa: PGH003
            t,  # type: ignore  # noqa: PGH003
            latents,  # type: ignore  # noqa: PGH003
            return_dict=False,
        )[0]

    latents = (latents / vae_scaling_factor) + vae_shift_factor

    image = vae.decode(latents, return_dict=False)[0]
    image = image_processor.postprocess(image, output_type="pt")
    assert isinstance(image, torch.Tensor)

    pil_images = image_processor.numpy_to_pil(image_processor.pt_to_numpy(image))
    pil_image = pil_images[0]

    pil_image.save(f"{input_hash}-{dtype}.png")
    if (np.array(reference_image) == np.array(pil_image)).all():
        logger.info("correct result")
    else:
        logger.error("wrong result")

    # processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    # torch_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

    # reference_model = SegformerForSemanticSegmentationReference(config=torch_model.config)
    # state_dict = torch_model.state_dict()

    # new_state_dict = {}
    # keys = [name for name, parameter in reference_model.state_dict().items()]
    # values = [parameter for name, parameter in state_dict.items()]
    # for i in range(len(keys)):
    #     new_state_dict[keys[i]] = values[i]

    # reference_model.load_state_dict(new_state_dict)
    # reference_model.eval()

    # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # image = Image.open(requests.get(url, stream=True).raw)
    # inputs = processor(images=image, return_tensors="pt")

    # torch_output = reference_model(inputs.pixel_values)

    # parameters = preprocess_model_parameters(
    #     initialize_model=lambda: reference_model, custom_preprocessor=create_custom_preprocessor(device), device=None
    # )
    # parameters = move_to_device(parameters, device)

    # for i in range(4):
    #     parameters["decode_head"]["linear_c"][i]["proj"]["weight"] = ttnn.to_device(
    #         parameters["decode_head"]["linear_c"][i]["proj"]["weight"], device=device
    #     )
    #     parameters["decode_head"]["linear_c"][i]["proj"]["bias"] = ttnn.to_device(
    #         parameters["decode_head"]["linear_c"][i]["proj"]["bias"], device=device
    #     )

    # ttnn_model = TtSegformerForSemanticSegmentation(torch_model.config, parameters)

    # torch_input_tensor_permuted = torch.permute(inputs.pixel_values, (0, 2, 3, 1))
    # ttnn_input_tensor = ttnn.from_torch(
    #     torch_input_tensor_permuted,
    #     dtype=ttnn.bfloat16,
    #     memory_config=ttnn.L1_MEMORY_CONFIG,
    #     device=device,
    #     layout=ttnn.TILE_LAYOUT,
    # )

    # ttnn_output = ttnn_model(
    #     ttnn_input_tensor,
    #     output_attentions=None,
    #     output_hidden_states=None,
    #     return_dict=None,
    #     parameters=parameters,
    # )

    # ttnn_output = ttnn.to_torch(ttnn_output.logits)
    # ttnn_output = torch.permute(ttnn_output, (0, 3, 1, 2))
    # h = w = int(math.sqrt(ttnn_output.shape[-1]))
    # ttnn_final_output = torch.reshape(ttnn_output, (ttnn_output.shape[0], ttnn_output.shape[1], h, w))

    # assert_with_pcc(torch_output.logits, ttnn_final_output, pcc=0.985)


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py
def _get_clip_prompt_embeds(
    prompt: list[str],
    *,
    num_images_per_prompt: int,
    device: torch.device | str,
    tokenizer_max_length: int,
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModelWithProjection,
) -> tuple[torch.Tensor, torch.Tensor]:
    prompt_count = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer_max_length,
        truncation=True,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids
    untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1]:
        logger.warning("CLIP input text was truncated")

    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)
    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]

    _, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(prompt_count * num_images_per_prompt, seq_len, -1)

    pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt, 1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(prompt_count * num_images_per_prompt, -1)

    return prompt_embeds, pooled_prompt_embeds


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py
def _get_t5_prompt_embeds(
    prompt: list[str],
    *,
    num_images_per_prompt: int,
    max_sequence_length: int,
    device: torch.device | str,
    tokenizer: T5TokenizerFast,
    text_encoder: T5EncoderModel,
) -> torch.Tensor:
    prompt_count = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1]:
        logger.warning("CLIP input text was truncated")

    prompt_embeds = text_encoder.forward(text_input_ids.to(device))[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    return prompt_embeds.view(prompt_count * num_images_per_prompt, seq_len, -1)


def main() -> None:
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    torch.set_grad_enabled(False)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    run(device=device)


main()
