# FLUX.3 [klein]

## Introduction

[FLUX.3 Image klein](https://huggingface.co/black-forest-labs/flux-3-image-klein) is Black Forest
Labs' 5.6B-parameter open-weights image model for text-to-image generation and editing with up to
10 reference images. The step-distilled checkpoint samples in 16 steps without classifier-free
guidance. It is released under the FLUX Kommunity License.

## Details

The model consists of a Qwen3-VL text encoder, a transformer and the FLUX.2 VAE.

**Text encoder.** The prompt goes through Qwen3-VL's chat template and is right-padded to a multiple
of 80 tokens. The embedding of each token is the concatenation of the hidden states at the inputs of
layers 4, 8, …, 32 of the 36-layer language model, 8 × 2560 = 20480 wide. The transformer attends to
the padding, so its states are part of the conditioning; the padding tokens take position 1, as in
the training environment. The pipeline only builds the first 32 layers.

**Transformer.** Three token modes share one sequence: the noisy image tokens, the reference image
tokens and the text. Each mode first runs through five per-mode blocks with its own weights, in
which it attends only to itself. All modes then run through 28 joint blocks, which share one set of
weights, but modulate each mode on its own: the image tokens by the flow timestep, the text and the
reference tokens by timestep 0. The step-distilled checkpoint is a flow map: each step also embeds
the timestep it jumps to, and adds it to every mode's timestep embedding. Every block is a parallel
attention and SwiGLU block with QK RMS norm and 4-axis RoPE `(t, h, w, l)`. Image positions are
scaled by 0.5, reference image `k` sits at `l = -(k + 1)`, and text advances along `l`.

**VAE.** The FLUX.2 VAE: 8× downsampling to 32 channels, with 2×2 patches packed into 128-channel
tokens that are normalized by the VAE's batch norm statistics.

**Schedule.** Flow matching with a constant timestep shift of 6.93.

## Implementation

`models/transformers/transformer_flux3.py` builds on the FLUX.2 single-stream block: the attention's
output projection and the feedforward's second linear are one matmul over their concatenated inputs,
fused with the gated residual on ring topologies. All block kinds share `Flux3Block`; the joint
blocks pass the prompt and the image to the attention as two streams, so that each keeps its own
modulation without slicing the sequence. With reference images, the image and reference tokens form
one stream, modulated row by row.

The prompt is the attention's ring stream and the image its joint stream, the reverse of FLUX.2. Ring
joint SDPA garbles some outputs when the sharded joint stream ends in a partial tile, as an
80-token prompt on four devices does (16 real and 16 padding tokens on the third), but it masks a
partial tile of the ring stream correctly. Both orders take the same time.

The time-embedding projections are merged into two chunked matmuls, one per timestep embedding, with
the `+1` of each scale folded into the bias.

Activations are sharded on the sequence over the SP axis and on the hidden dimension over the TP
axis. The prompt is sharded over SP too, padded to whole tiles per device; the attention masks the
padding by its logical length, so the prompt can have any multiple of 80 tokens. The image and
reference token counts must be multiples of 32 × SP. The transformer requires SP > 1.

The pipeline (`pipelines/flux3/pipeline_flux3.py`) supports the step-distilled checkpoint. It traces
one denoising step per resolution, reference count and prompt length, and warms all of them up when
it is created. A reference image is center-cropped to the output's aspect ratio and resized to the
output size; diffusers instead keeps a single reference whole and generates at its size.

The text encoder is not traced: the multimodal RoPE of `Qwen3VlEncoder` writes its frequency table
to the device on every call, which trace capture rejects. It takes about 0.2 s per prompt.

## Performance

Traced, 16 steps, 1024 × 1024, on a T3000 (2 × 4 Wormhole, SP 4, TP 2):

| Mode              | Text encoder | Denoising      | VAE decode | Total  |
| ----------------- | ------------ | -------------- | ---------- | ------ |
| Text to image     | 0.2 s        | 5.1 s (0.32 s) | 0.8 s      | 6.3 s  |
| Edit, 1 reference | 0.2 s        | 9.0 s (0.56 s) | 1.0 s      | 10.5 s |

The denoising column gives the time per step in parentheses. The edit also encodes the reference
with the VAE, in 0.3 s.

## Accuracy

On 2 × 4 Wormhole, against the diffusers transformer in bf16 with random inputs, a 1024 × 1024
image and a 512 × 512 reference, the full transformer reaches PCC 99.6–99.7 % (RMSE 7.8–8.3 % of
σ) for text to image and 99.4–99.6 % (RMSE 9.4–11.4 %) for editing, for prompts of 80 to 640
tokens. The diffusers model itself differs by RMSE 5–6 % between bf16 and float32. End to end, a
16-step generation matches the torch reference with pixel PCC 93 %, and the two images agree in
composition and text.

## Tests

- `tests/models/flux3/test_transformer_flux3.py`: the transformer against the diffusers reference,
  vendored in `tests/models/flux3/reference_flux3.py` until the venv's diffusers has FLUX.3.
- `tests/models/flux3/test_text_encoder_flux3.py`: the text embeddings against transformers.
- `tests/models/flux3/test_pipeline_flux3.py`: generation, and an end-to-end comparison with a torch
  reference of diffusers' distilled pipeline.
