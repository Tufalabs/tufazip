# tufazip v2

Deterministic decoder for the 100,958,324-byte `enwik9-v2.bin` artifact.

## Run

Requirements: Linux x86-64, exactly two visible NVIDIA B200 GPUs, GCC, CUDA
13.0, cuDNN 9.20, Python 3.12.3, and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --frozen --python 3.12.3
uv run --frozen --python 3.12.3 ./tufazip /path/to/enwik9-v2.bin /path/to/enwik9 /path/to/state
```

The state directory holds the rebuilt prior (see below) and resumable
checkpoints. If the process is interrupted, repeat the identical command.

Expected files:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `enwik9-v2.bin` | 100,958,324 | `198451a21f64386b487b8ea97e141562db355eec9cc9af34e7bbb6a022bcce77` |
| `weights.tz` | 4,147,476 | `ab55ebd0d7d63e3c2b6731489a9f8c2e03bb18eb67d8bc5b39325d34d100199a` |
| rebuilt `weights.bin` | 4,334,741 | `e94c089f8eb985a87307dca3a7702385fcaccd14d95743811f8788f7da24448d` |
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

## Smaller decoder package

The decoder now ships a losslessly recompressed prior and only the code it
runs. The bitstream, model and online learning are unchanged.

| Decoder ZIP part | verified archive | this package |
| --- | ---: | ---: |
| Prior weights (`weights.bin` → `weights.tz`) | 4,322,949 | 4,147,476 |
| Detokenizer (`preprocess.c` → `detok.c`) | 7,541 | 1,446 |
| Python sources (comments and docstrings stripped) | 29,160 | 23,605 |
| Everything else (vocab, lock, licences, README, ZIP headers) | 70,379 | 69,836 |
| **ZIP total** | **4,430,029** | **4,242,363** |

- **`weights.tz`** rebuilds the exact bytes of `weights.bin`, and the launcher
  checks the SHA-256 above before decoding. Most of the prior is 1-bit signs,
  and their columns are biased: `head.weight` carries only 0.80 bits per sign.
  Each sign is arithmetic-coded with probability `squash(b[j] + u_i·V[j])`.
  `b` (column bias) and `V` (top singular directions, k ≤ 4) are sent as int8,
  and every row's `u_i` is learned online from that row's already-decoded
  signs. Matrices that would not shrink stay raw. The fp16 row scales use an
  adaptive bitwise tree, and the metadata and 8-bit norms use LZMA. The model
  is integer-only, so it gives identical results on any CPU. Unpacking takes
  about 4 s. `tools/wpack.py` is the encoder; it is not needed to decode.
- **`detok.c`** is the decode half of the NNCP preprocessor. It is
  byte-identical to `preprocess d` on random and edge-case token streams.
- **Python sources** in the ZIP come from `tools/make_zip.py`, which strips
  comments, docstrings and blank lines. It refuses any result whose AST differs
  from the original's (minus docstrings). `nn.py` changes only in a
  size-accounting list that never affects predictions.

Build it with `python3 tools/make_zip.py` (writes `dist/decoder.zip`). With
this package the ranked total would be `100,958,324 + 4,242,363 = 105,200,687` bytes, 187,666 bytes
less than above. The weight round trip, the detokenizer, AST equivalence and a
single-GPU encode→decode with the stripped decoder were all checked. The full
two-B200 decode of `enwik9-v2.bin` with this package has not been re-run, so the
submission data above still describes the verified archive.

v1, with its original decoder and run instructions, is available at
[commit 794dda3](https://github.com/Tufalabs/tufazip/tree/794dda3e097957619177f83baf20a4fd0d33b51f).

## Acknowledgements

- Thanks to Jerome Sieber for providing mentorship and advising for this project and Tufa Labs for the providing the compute and capacity to work on this project!
- Thanks to Matt Mahoney for creating and maintaining the Large Text Compression Benchmark (https://www.mattmahoney.net/dc/text.html) still after 20 years! as well as his online textbook Data Compression Explained (https://mattmahoney.net/dc/dce.html)!
