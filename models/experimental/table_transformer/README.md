# Table Transformer Demo

## Introduction

[Table Transformer](https://github.com/microsoft/table-transformer) is a neural network, based on the transformer architecture, for detecting tables in images and recognizing table structure. The main components of the model are a ResNet backbone and a modified transformer with positional encoding.

## Details

The model architecture is described in the paper
[End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872).
To summarize, the model consists of the following components:

- **Backbone**: ResNet-18 with the final pooling and feed-forward layers removed to preserve spatial structure.
- **Input projection**: 1x1 convolution to reduce dimensionality.
- **Noncausal pre-LN transformer** with two kinds of positional encodings:
  - Sine positional encoding added to the input of each attention layer in the encoder and to the cross-attention layers in the decoder.
  - Learned positional encoding added to the input of each attention layer in the decoder.
- **Linear classification layer**.
- **Location prediction**: Multilayer perceptron with ReLU activation function and a final sigmoid applied to the output.

## Implementation Status

`TableTransformer` is currently implemented as a PyTorch module to make use of `load_state_dict` for loading the model checkpoint. No memory optimizations are implemented yet. All tensors are tiled, interleaved, reside on DRAM, and have data type `float32`. The following are still implemented in PyTorch:

- Flattening of the encoder input.
- Generation of sine positional encoding.
- Generation of the decoder input.

The model contains the following modules:

- `ResNet`: Not yet implemented using ttnn; uses PyTorch implementation.
- `Linear`: Implemented using ttnn.
- `Transformer`: Implemented using ttnn.
  - `TransformerEncoderLayer`: Implemented using ttnn.
  - `TransformerDecoderLayer`: Implemented using ttnn.
  - `Attention`: Mostly implemented using ttnn. Pads the query tensor, which is not ideal. Final removal of the padding is done using PyTorch.
  - `LayerNorm`: Implemented using ttnn.
- `MultilayerPerceptron`: Implemented using ttnn.

## Running the Demo

The demo is run using the following command:

```sh
pytest models/experimental/table_transformer/demo/demo.py
```

This generates output files with the detected tables highlighted, as indicated by the log output.
