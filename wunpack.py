"""Rebuild the exact weights.bin bytes from weights.tz (encoder: tools/wpack.py).

Sign bits are coded column by column with a binary arithmetic coder. Each
column's probability is squash(b[j] + u_i . V[j]): b and V are transmitted, and
each row's u_i is learned online from that row's already-coded bits. The fp16
row scales use an adaptive bitwise tree; everything else is LZMA. All modelling
is integer-only, so encoder and decoder agree on every platform.
"""
import json
import lzma
import struct

import numpy as np

LZMA_FILTERS = [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}]
_KNOTS = (1, 2, 3, 6, 10, 16, 27, 45, 73, 120, 194, 310, 488, 747, 1101, 1546,
          2047, 2549, 2994, 3348, 3607, 3785, 3901, 3975, 4022, 4050, 4068,
          4079, 4085, 4089, 4092, 4093, 4094)


def _squash(d):
    w, i = d & 127, (d >> 7) + 16
    return (_KNOTS[i] * (128 - w) + _KNOTS[i + 1] * w + 64) >> 7


SQUASH = np.array([_squash(d) for d in range(-2047, 2048)], dtype=np.int64)


class Decoder:
    def __init__(self, data):
        self.data, self.pos = data, 4
        self.x1, self.x2, self.x = 0, 0xFFFFFFFF, int.from_bytes(data[:4], "big")

    def bits(self, probs, _=None):
        """Decode one bit per 12-bit probability of a one."""
        data, pos, end = self.data, self.pos, len(self.data)
        x1, x2, x = self.x1, self.x2, self.x
        out = []
        for p in probs:
            xmid = x1 + ((x2 - x1) * p >> 12)
            y = x <= xmid
            if y:
                x2 = xmid
            else:
                x1 = xmid + 1
            out.append(y)
            while not (x1 ^ x2) & 0xFF000000:
                x1 = x1 << 8 & 0xFFFFFFFF
                x2 = (x2 << 8 & 0xFFFFFFFF) | 255
                x = (x << 8 & 0xFFFFFFFF) | (data[pos] if pos < end else 255)
                pos += 1
        self.pos, self.x1, self.x2, self.x = pos, x1, x2, x
        return out


def code_scales(coder, count, values=None):
    """Adaptive bitwise tree over one tensor's raw fp16 bit patterns."""
    probs, out = {}, []
    for n in range(count):
        node = 1
        for bit in range(15, -1, -1):
            key = node if bit > 5 else -bit
            p = probs.get(key, 2048)
            y = coder.bits([p], values and [values[n] >> bit & 1])[0]
            p += (4096 - p) >> 5 if y else -(p >> 5)
            probs[key] = min(4095, max(1, p))
            node = 2 * node + y
        out.append(node & 0xFFFF)
    return np.array(out, dtype="<u2")


def code_signs(coder, rows, cols, b, V, lr, signs=None):
    """Code a rows x cols {0,1} matrix in column order; returns it."""
    k = V.shape[1]
    U = np.zeros((rows, k), dtype=np.int64)
    out = np.empty((cols, rows), dtype=np.int64)
    for j in range(cols):
        z = b[j] + ((U @ V[j]) >> 13) if k else np.full(rows, b[j])
        p = SQUASH[np.clip(z, -2047, 2047) + 2047]
        y = np.array(coder.bits(p.tolist(), None if signs is None else
                                signs[:, j].tolist()), dtype=np.int64)
        if k:
            U += (((y << 12) - p)[:, None] * V[j] * lr) >> 10
        out[j] = y
    return out.T


def unpack(src, dst):
    data = open(src, "rb").read()
    side_len, = struct.unpack("<I", data[:4])
    side = lzma.decompress(data[4:4 + side_len], format=lzma.FORMAT_RAW,
                           filters=LZMA_FILTERS)
    header_len, plan_len = struct.unpack("<II", side[:8])
    header = side[8:8 + header_len]
    plan = json.loads(side[8 + header_len:8 + header_len + plan_len])
    side_pos = 8 + header_len + plan_len
    entries = json.loads(header[12:])["entries"]
    raw_pos = 4 + side_len
    raw_len = sum(e["value_bytes"] for e, mode in zip(entries, plan["modes"])
                  if mode == "raw")
    coder = Decoder(data[raw_pos + raw_len:])
    scales = [code_scales(coder, e["scale_count"]) for e in entries]
    parts = [header]
    for entry, mode, scale in zip(entries, plan["modes"], scales):
        parts.append(scale.tobytes())
        size = entry["value_bytes"]
        if mode == "raw":
            parts.append(data[raw_pos:raw_pos + size])
            raw_pos += size
        elif mode == "byte":
            parts.append(side[side_pos:side_pos + size])
            side_pos += size
        else:
            rows, cols = entry["shape"]
            k, lr, bshift = mode
            b = np.frombuffer(side, np.int8, cols, side_pos).astype(np.int64)
            side_pos += cols
            V = np.frombuffer(side, np.int8, cols * k, side_pos)
            side_pos += cols * k
            signs = code_signs(coder, rows, cols, b << bshift,
                               V.reshape(cols, k).astype(np.int64), lr)
            parts.append(np.packbits(signs.astype(np.uint8),
                                     bitorder="little").tobytes())
    with open(dst, "wb") as out:
        out.write(b"".join(parts))


if __name__ == "__main__":
    import sys
    unpack(sys.argv[1], sys.argv[2])
