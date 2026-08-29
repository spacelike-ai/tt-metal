# Model performance and accuracy

Performance collected from [demo/demo.py](demo/demo.py) with the `batch-1` test case.

## Configuration

| Parameter | Text Performance | Vision Performance |
|-----------|------------------|-------------------|
| Batch size | 1 | 1 |
| Prompt file | `demo/sample_prompts/text_only.json` | `demo/sample_prompts/demo.json` |
| Image resolution | N/A | 2048×1365 pixels (demo.jpeg) |
| Vision tokens | N/A | ~2752 tokens (86×128 patches, 16×16 patch size, 2×2 spatial merge) |
| Input sequence length (ISL) | Short text prompt | ~2752 vision tokens + text prompt |
| Output sequence length (OSL) | Up to 200 tokens | Up to 200 tokens |
| Max sequence length | 4096 | 4096 |
| Precision | bfp4 MLP, bfp8 attention | bfp4 MLP, bfp8 attention |

## Text Performance

| Model             | Device      | Speed (t/s/u) | TTFT (ms) |
|-------------------|-------------|---------------|-----------|
| qwen3_vl-32b    | T3K          | 18.72         | 1828.41   |
| fibo-vlm        | T3K          | 40.40         | 784.28    |
| fibo-vlm        | N300         | 31.74         | 715.68    |
| fibo-vlm        | N150         | 23.59         | 117.82    |

## Vision Performance

| Model             | Device      | Speed (t/s/u) | TTFT (ms) |
|-------------------|-------------|---------------|-----------|
| qwen3_vl-32b    | T3K          | 1148.41        | 2400     |
| fibo-vlm        | T3K          | 2149.66        | 825.55   |
| fibo-vlm        | N300         | 2128.73        | 733.29   |
| fibo-vlm        | N150         | 2372.21        | 878.46   |
