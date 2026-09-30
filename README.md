# tufazip v2

Blog: https://tufalabs.ai/research/tufazip

Tweet: https://x.com/tufalabs/status/2123350647273631907

Deterministic decoder for the 100,958,324-byte `enwik9-v2.bin` artifact.

## Run

Requirements: Linux x86-64, exactly two visible NVIDIA B200 GPUs, GCC, CUDA
13.0, cuDNN 9.20, Python 3.12.3, and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --frozen --python 3.12.3
uv run --frozen --python 3.12.3 ./tufazip /path/to/enwik9-v2.bin /path/to/enwik9 /path/to/state
```

The state directory is optional and only stores resumable checkpoints. If the
process is interrupted, repeat the identical command.

Expected files:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `enwik9-v2.bin` | 100,958,324 | `198451a21f64386b487b8ea97e141562db355eec9cc9af34e7bbb6a022bcce77` |
| `weights.bin` | 4,334,741 | `e94c089f8eb985a87307dca3a7702385fcaccd14d95743811f8788f7da24448d` |
| decoded `enwik9` | 1,000,000,000 | `159b85351e5f76e60cbe32e04c677847a9ecba3adc79addab6f4c6c7aa3744bc` |

## Submission data

| Field | Value |
| --- | --- |
| Program and version | `tufazip v2` |
| Author | Tommy He |
| Repository | https://github.com/Tufalabs/tufazip |
| Compressed enwik8 | - |
| Compressed enwik9 | 100,958,324 bytes |
| Decompressor ZIP | 4,430,029 bytes |
| Ranked total | 105,388,353 bytes |
| Compression time | ~ 631,548 seconds, excluding offline training |
| Decompression time | ~ 645,748 seconds |
| Machine | 2× NVIDIA B200 GPUs; Linux x86-64 |
| Runtime | Python 3.12.3; NumPy 2.4.6; PyTorch 2.12.0+cu130; CUDA 13.0; cuDNN 9.20 |
| Algorithm | Online Transformer initialized from a quantized offline Cycle2 prior |

The reported ZIP is the original verified decoder archive, including the
weights and vocabulary. The ranked total is
`100,958,324 + 4,430,029 = 105,388,353` bytes.

v1, with its original decoder and run instructions, is available at
[commit 794dda3](https://github.com/Tufalabs/tufazip/tree/794dda3e097957619177f83baf20a4fd0d33b51f).

## Acknowledgements

- Thanks to Jerome Sieber for providing mentorship and advising for this project and Tufa Labs for the providing the compute and capacity to work on this project!
- Thanks to Matt Mahoney for creating and maintaining the Large Text Compression Benchmark (https://www.mattmahoney.net/dc/text.html) still after 20 years! as well as his online textbook Data Compression Explained (https://mattmahoney.net/dc/dce.html)!
