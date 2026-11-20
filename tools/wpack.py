#!/usr/bin/env python3
"""Losslessly recompress weights.bin into weights.tz (inverse: wunpack.py).

Only the encoder lives here; the shipped decoder is wunpack.py. Both drive the
same model code (wunpack.code_signs / code_scales), so they cannot drift apart.
The result is verified by decoding it and comparing SHA-256 with the input.
"""
import argparse
import hashlib
import json
import lzma
import multiprocessing
import os
import struct
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import wunpack as W  # noqa: E402

NEG_LOG2 = -np.log2(np.maximum(np.arange(4097), 1e-300) / 4096.0)
CANDIDATE_K = (0, 2, 4, 8)
CANDIDATE_LR = (4, 6, 10, 16)
BSHIFT = 4


class Encoder:
    def __init__(self):
        self.x1, self.x2, self.out = 0, 0xFFFFFFFF, bytearray()

    def bits(self, probs, ys):
        x1, x2, out = self.x1, self.x2, self.out
        for p, y in zip(probs, ys):
            xmid = x1 + ((x2 - x1) * p >> 12)
            if y:
                x2 = xmid
            else:
                x1 = xmid + 1
            while not (x1 ^ x2) & 0xFF000000:
                out.append(x2 >> 24)
                x1 = x1 << 8 & 0xFFFFFFFF
                x2 = (x2 << 8 & 0xFFFFFFFF) | 255
        self.x1, self.x2 = x1, x2
        return ys

    def finish(self):
        # Decoder pads with 0xFF, so x1's top byte lands inside [x1, x2].
        self.out.append(self.x1 >> 24)
        return bytes(self.out)


def read_archive(path):
    data = open(path, "rb").read()
    meta_len, = struct.unpack("<Q", data[4:12])
    header = data[:12 + meta_len]
    entries = json.loads(data[12:12 + meta_len])["entries"]
    pos, tensors = len(header), []
    for e in entries:
        scales = np.frombuffer(data, "<u2", e["scale_count"], pos)
        pos += e["scale_bytes"]
        values = data[pos:pos + e["value_bytes"]]
        pos += e["value_bytes"]
        tensors.append((e, scales, values))
    assert pos == len(data), "trailing bytes"
    return header, tensors


def signs_of(entry, values):
    rows, cols = entry["shape"]
    bits = np.unpackbits(np.frombuffer(values, np.uint8), bitorder="little")
    return bits[:rows * cols].reshape(rows, cols).astype(np.int64)


def side_params(signs, k, lr):
    rows, cols = signs.shape
    p = (signs.sum(0) + 0.5) / (rows + 1.0)
    b = np.clip(np.round(np.log(p / (1 - p)) * 256 / (1 << BSHIFT)), -128, 127)
    if k:
        s = 2.0 * signs - 1
        s -= s.mean(0)
        _, _, vt = np.linalg.svd(s, full_matrices=False)
        V = np.clip(np.round(vt[:k].T * np.sqrt(cols) * 32), -127, 127)
    else:
        V = np.zeros((cols, 0))
    return b.astype(np.int8), V.astype(np.int8)


def model_cost(signs, b, V, lr):
    """Ideal code length in bytes of wunpack.code_signs (same integer model)."""
    rows, cols = signs.shape
    k = V.shape[1]
    b = b.astype(np.int64) << BSHIFT
    V = V.astype(np.int64)
    U = np.zeros((rows, k), dtype=np.int64)
    total = 0.0
    for j in range(cols):
        z = b[j] + ((U @ V[j]) >> 13) if k else np.full(rows, b[j])
        p = W.SQUASH[np.clip(z, -2047, 2047) + 2047]
        y = signs[:, j]
        total += NEG_LOG2[np.where(y == 1, p, 4096 - p)].sum()
        if k:
            U += (((y << 12) - p)[:, None] * V[j] * lr) >> 10
    return total / 8


def lz_size(raw):
    return len(lzma.compress(raw, format=lzma.FORMAT_RAW, filters=W.LZMA_FILTERS))


_SIGNS = {}


def evaluate(job):
    index, k, lr = job
    signs = _SIGNS[index]
    b, V = side_params(signs, k, lr)
    return index, k, lr, model_cost(signs, b, V, lr) + lz_size(b.tobytes() + V.tobytes())


def choose_modes(tensors, workers):
    modes = []
    for index, (e, _, values) in enumerate(tensors):
        if e["bits"] == 1 and len(e["shape"]) == 2:
            _SIGNS[index] = signs_of(e, values)
            modes.append("raw")
        else:
            modes.append("byte")
    jobs = [(i, k, lr) for i in _SIGNS for k in CANDIDATE_K
            for lr in (CANDIDATE_LR if k else (0,))]
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        results = pool.map(evaluate, jobs)
    for index in _SIGNS:
        e = tensors[index][0]
        best = min((r for r in results if r[0] == index), key=lambda r: r[3])
        saving = e["value_bytes"] - best[3]
        if saving > 64:
            modes[index] = [best[1], best[2], BSHIFT]
        print("{:28s} raw={:8d} model={:10.0f} k={} lr={:2d} -> {}".format(
            e["name"], e["value_bytes"], best[3], best[1], best[2],
            modes[index]), flush=True)
    return modes


def pack(src, dst, workers):
    header, tensors = read_archive(src)
    modes = choose_modes(tensors, workers)
    plan = json.dumps({"modes": modes}, separators=(",", ":")).encode()
    side, raw, coder = [struct.pack("<II", len(header), len(plan)), header, plan], [], Encoder()
    for e, scales, values in tensors:
        W.code_scales(coder, e["scale_count"], [int(v) for v in scales])
    for (e, _, values), mode in zip(tensors, modes):
        if mode == "raw":
            raw.append(values)
        elif mode == "byte":
            side.append(values)
        else:
            k, lr, _ = mode
            signs = signs_of(e, values)
            b, V = side_params(signs, k, lr)
            side += [b.tobytes(), V.tobytes()]
            W.code_signs(coder, *signs.shape, b.astype(np.int64) << BSHIFT,
                         V.astype(np.int64), lr, signs)
    side_lz = lzma.compress(b"".join(side), format=lzma.FORMAT_RAW,
                            filters=W.LZMA_FILTERS)
    with open(dst, "wb") as out:
        out.write(struct.pack("<I", len(side_lz)) + side_lz + b"".join(raw)
                  + coder.finish())


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", nargs="?", default=os.path.join(ROOT, "weights.bin"))
    parser.add_argument("dst", nargs="?", default=os.path.join(ROOT, "weights.tz"))
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()
    pack(args.src, args.dst, args.workers)
    with tempfile.TemporaryDirectory() as tmp:
        check = os.path.join(tmp, "weights.bin")
        W.unpack(args.dst, check)
        if sha256(check) != sha256(args.src):
            sys.exit("round trip FAILED")
    print("{} -> {}: {} -> {} bytes, round trip verified".format(
        args.src, args.dst, os.path.getsize(args.src), os.path.getsize(args.dst)))


if __name__ == "__main__":
    main()
