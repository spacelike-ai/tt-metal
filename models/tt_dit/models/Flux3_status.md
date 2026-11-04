# FLUX.3 [klein] — Status

Status of the FLUX.3 [klein] port as of 2026-11-04. The architecture and implementation are
described in [`Flux3.md`](Flux3.md).

## Summary

Text-to-image generation and editing with one reference image work on a T3000 (2 × 4 Wormhole) and
match the diffusers reference. Only the step-distilled checkpoint is supported. No existing file was
changed; the port is all new files.

| Piece                                  | Status                                                |
| -------------------------------------- | ----------------------------------------------------- |
| Transformer                            | Passes on 2 × 4, both layouts, 24 cases               |
| Text encoder (Qwen3-VL, 32 layers)     | Passes on 2 × 4, untraced only                        |
| Pipeline, text to image                | Works, traced; matches the torch reference end to end |
| Pipeline, edit with one reference      | Works, traced; needs `Flux2VaeEncoder` (see below)    |
| 2 × 2 and 4 × 8 Blackhole, 4 × 8 Galaxy | Presets written, not run                              |

## Files

- `models/transformers/transformer_flux3.py` — the transformer and `Flux3Checkpoint`.
- `pipelines/flux3/text_encoder.py` — the Qwen3-VL text encoder wrapper.
- `pipelines/flux3/pipeline_flux3.py` — the pipeline, with mesh presets.
- `tests/models/flux3/` — tests, device parameters, and `reference_flux3.py`, a copy of the
  diffusers FLUX.3 transformer, because the venv's diffusers 0.38 does not have it.

## Test results

All on `loud` (T3000, 2 × 4 Wormhole).

`test_transformer_flux3.py -k 2x4`: 24 passed. Against the diffusers transformer in bf16 with random
inputs, 1024 × 1024 image, 512 × 512 reference:

| Case                  | PCC          | RMSE / σ    |
| --------------------- | ------------ | ----------- |
| Few blocks            | 99.91–99.94 % | 1.4–1.6 %  |
| All blocks, t2i       | 99.67–99.71 % | 7.8–8.3 %  |
| All blocks, edit      | 99.37–99.57 % | 9.4–11.4 % |

For scale, the diffusers transformer itself differs by RMSE 5–6 % between bf16 and float32.

`test_text_encoder_flux3.py -k 2x4`: 2 passed. Per output layer PCC ≥ 99.91 % against transformers.

`test_pipeline_flux3.py::test_pipeline_accuracy -k 2x4`: passed. A 16-step generation against a
torch reference of diffusers' distilled pipeline reaches pixel PCC 93 %; the images agree in
composition and text.

`test_pipeline_flux3.py::test_pipeline -k "2x4 and -traced"` with `NO_PROMPT=1`: passed; the images
are as prompted, and the edit changes what it is asked to.

Performance, traced, 1024 × 1024, 16 steps on 2 × 4 Wormhole:

| Mode              | Text encoder | Denoising (per step) | VAE decode | Total  |
| ----------------- | ------------ | -------------------- | ---------- | ------ |
| Text to image     | 0.2 s        | 5.1 s (0.32 s)       | 0.8 s      | 6.3 s  |
| Edit, 1 reference | 0.2 s        | 9.0 s (0.56 s)       | 1.0 s      | 10.5 s |

## Dependencies on uncommitted changes

The port was developed and tested in a working tree with other uncommitted changes. Two of them
matter:

1. `models/vae/vae_flux2.py` — `Flux2VaeEncoder`, which the pipeline uses to encode reference
   images. Without it, the pipeline cannot be imported.
2. `blocks/attention_opt.py` — the ring SDPA core grid computed per call instead of once from the
   first call's sequence length. Committed `attention_opt.py` fixes the grid from the first call;
   FLUX.3 calls the same attention with several prompt lengths, so it was only tested with the
   per-call grid.

## Issues found

### 1. Ring joint SDPA garbles rows when the joint stream ends in a partial tile

`ttnn.transformer.ring_joint_scaled_dot_product_attention` with a sharded joint stream
(`persistent_output_buffer_joint_k/v`) and a `logical_l` that is not a multiple of 32 computes some
output rows wrongly. With 4096 image tokens on 4 devices and an 80-token prompt padded to 128, a few
scattered rows of both outputs have errors above 100 %, while every row stays below 15 % at 96 or
128 tokens. The padding values do not matter: filling the padding with large random values gives
bit-identical outputs, so padding is masked; real keys seem to be masked or mishandled instead.

A sweep of joint lengths (whole-output PCC of the joint output; fine is ≥ 0.9995):

| Padded L | Logical L | SP 4    | SP 2    |
| -------- | --------- | ------- | ------- |
| 128      | 80        | 0.99966 | 0.99973 |
| 128      | 96        | 0.99980 | 0.99980 |
| 128      | 112       | 0.98543 | 0.99154 |
| 128      | 128       | 0.99986 | 0.99986 |
| 256      | 176       | 0.99433 | 0.99659 |
| 256      | 208       | 0.99959 | 0.99969 |
| 256      | 240       | 0.99402 | 0.99782 |

The whole-output PCC hides the 80-token case (the spatial output there is at PCC 0.9974), which is
why `test_sdpa_partial_joint_tile.py` checks the largest per-row error instead. The existing
sharded-prompt tests in `tests/unit/test_ring_joint_attention.py` do not cover these lengths.

In the full FLUX.3 transformer this raised the output error at 80 tokens from 7.8 % to 14.9 % RMSE
(PCC 98.9 %, below the test's 99 %), consistently across seeds, and likewise at 112 tokens.

Workaround: `Flux3Block` passes the prompt as the attention's ring stream and the image as the
joint stream, the reverse of FLUX.2. The ring stream's partial tile (`logical_n`) is masked
correctly, the image is a whole number of tiles, and both orders take the same time.

Follow-ups:

- Fix the kernel; `tests/models/flux3/test_sdpa_partial_joint_tile.py` reproduces the bug and its
  two failing cases are marked `xfail`.
- Check FLUX.2, which passes the prompt as the joint stream with the same padding scheme.

### 2. The Qwen3-VL text encoder cannot be traced

`blocks/rope.py` `_inverse_frequencies` calls `ttnn.arange(..., device=device)` on every forward,
which writes to the device; trace capture fails with "Writes are not supported during trace
capture". So the FLUX.3 text encoder runs untraced, about 0.2 s per prompt. Caching the frequency
table in `Rope` would allow tracing every `TransformerEncoder` with RoPE.

### 3. Some edit prompts barely change the image

"Turn the scene into a snowy winter night" returns an almost unchanged copy of the reference. The
torch reference does the same, and diffusers prepares edits exactly as the pipeline does, so this is
the model's behavior. Concrete, local edits work ("Make the letters on the sign blue and add snow on
the ledge").

## Limitations

- Only the step-distilled (flow-map) checkpoint; the base checkpoint needs classifier-free guidance,
  which is not implemented. The pipeline raises `NotImplementedError` for it.
- Sequence parallelism must be greater than 1.
- One prompt per call, as in diffusers.
- A reference image is center-cropped to the output's aspect ratio and resized to the output size.
  diffusers instead keeps a single reference whole and generates at its size.
- At most `max_references` references (default 1), each traced combination warmed up at creation.

## Next steps

Changes to existing files:

- Commit or rebase onto `Flux2VaeEncoder` in `models/vae/vae_flux2.py` and the per-call ring SDPA
  grid in `blocks/attention_opt.py`.
- Fix the ring joint SDPA partial-tile bug; then the stream swap in `Flux3Block` can go.
- Cache the RoPE frequency table so the text encoder can be traced.
- Add CI entries (`tests/pipeline_reorg/models_e2e_tests.yaml`) and the model to `testing.md` and
  `AGENTS.md`.
- Optionally, bump diffusers to a version with FLUX.3 and drop `reference_flux3.py`.
- Integrate with tt-inference-server.

Testing:

- Run on 2 × 2 and 4 × 8 Blackhole and on a Galaxy (via CI dispatch).
- Edit accuracy against the torch reference as a test (`_torch_reference` already takes a
  reference image).
