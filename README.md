# tufazip

Deterministic decoder for the 105,924,360-byte `enwik9.bin` artifact.

## Run

Requirements: Linux x86-64, exactly two visible NVIDIA B200 GPUs, GCC, CUDA
13.0, cuDNN 9.20, and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --frozen
uv run ./tufazip /path/to/enwik9.bin /path/to/enwik9 /path/to/state
```

The state directory is optional and only stores resumable checkpoints. If the
process is interrupted, repeat the identical command.

Expected files:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `enwik9.bin` | 105,924,360 | `a61841fcaac31d9db9b70dab797ac43f7502fc526326e1e9a326bdde8d24d01b` |
| decoded `enwik9` | 1,000,000,000 | `159b85351e5f76e60cbe32e04c677847a9ecba3adc79addab6f4c6c7aa3744bc` |

## Submission data

| Field | Value |
| --- | --- |
| Program and version | `tufazip 0.1.0` |
| Author | Tommy He |
| Public download | https://github.com/Tufalabs/tufazip |
| Compressed enwik8 | - |
| Compressed enwik9 | 105,924,360 bytes |
| Decompressor ZIP | 54,709 bytes |
| Ranked total | **105,979,069 bytes** |
| Compression time | ~ 647,595 seconds |
| Decompression time | ~ 630,136 seconds |
| Peak compression memory | 23.55 GiB host + 43.37 GiB aggregate GPU; ~ 68,525 MiB total |
| Peak decompression memory | 28.65 GiB host + 43.37 GiB aggregate GPU; ~ 73,748 MiB total |
| Machine | NVIDIA DGX B200; 2× B200 GPUs; 2× Intel Xeon Platinum 8570; Linux x86-64 |
| Runtime | Python 3.12.3; NumPy 2.4.6; PyTorch 2.12.0+cu130; CUDA 13.0; cuDNN 9.20; driver 580.159.04 |
| Algorithm | Online Transformer |

The reported ZIP is an Info-ZIP `zip -9 -X` archive containing only the files
needed to install and run the decompressor:

```bash
zip -9 -X tufazip-decompressor.zip \
  tufazip nn.py arithmetic_coder.py checkpoint_utils.py \
  preprocess.c cutils.h Makefile enwik9.voc pyproject.toml uv.lock
unzip -t tufazip-decompressor.zip
wc -c tufazip-decompressor.zip
```

The ranked total is `105,924,360 + 54,709 = 105,979,069` bytes.

## Acknowledgements
- Thanks to Jerome Sieber for providing mentorship and advising for this project and Tufa Labs for the providing the compute and capacity to work on this project!
- Thanks to Matt Mahoney for creating and maintaining the Large Text Compression Benchmark (https://www.mattmahoney.net/dc/text.html) still after 20 years! as well as his online textbook Data Compression Explained (https://mattmahoney.net/dc/dce.html)!
