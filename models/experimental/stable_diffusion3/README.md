# Stable Diffusion 3.5

## Introduction

[Stable Diffusion 3.5](https://stability.ai/news/introducing-stable-diffusion-3-5) is a generative model for image synthesis guided by text prompts.

## Details

The architecture is described in the paper
[Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206).

The model consists of two different text encoders together with their tokenizers, a scheduler, a trasformer and a VAE. The core component is the transformer, called MMDiT (Multimodal Diffusion Transformer). The transformer is made up of spatial, prompt and time embeddings, and a series of transformer blocks. Transformer blocks mainly contain attention layers, that operate either on the spatial embedding only, or on the spatial and prompt embeddings together.

## Implementation Status

- All operations of MMDiT are implemented using `ttnn` with the exception of linear transformations of the time embedding in the transformer blocks and at the end of the transformer. Using the `ttnn` implementation gives unusable results, possibly due to numerical precision issues.
  - When multiplying two bfloat16 matrices with dimensions AxB and BxC with B = 10_000, only about 26 % of the resulting elements are within 10 % of the correct value. Using float32 about 94 % are withing this bound.
- Almost all tensors have data type bfloat16 and reside on DRAM.
- Enabling the program cache makes the model give incorrect results.
- The VAE, the scheduler, the text encoders and tokenizers are taken from the `diffusers` library.
  - An update of the `diffusers` library was required.
- The T5 text encoder takes several seconds to encode a prompt on the CPU. It could be ported to `ttnn` to improve performance.

## Running the Tests

The tests are run using the following command:

```sh
pytest models/experimental/stable_diffusion3/tests
```

## Running the Demo

The demo is run using the following command:

```sh
pytest models/experimental/stable_diffusion3/demo.py
```
