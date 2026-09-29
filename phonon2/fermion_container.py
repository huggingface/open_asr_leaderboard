"""Reader for `fermion-five-value-parakeet-v1` containers (the Phonon-2 weight file).

Exact inverse of the writer in the training tooling; the record layout is documented in the container header
(five-value trit planes + magnitude bits, intN tables with per-row scales, fp16 tensors).  The reader here is
self-contained."""
from __future__ import annotations

import json

import numpy as np

FORMAT = "fermion-five-value-parakeet-v1"


def _trits(buf: bytes, o: int, i: int) -> np.ndarray:
    rb = (i + 4) // 5
    x = np.frombuffer(buf, dtype=np.uint8, count=o * rb).reshape(o, rb).astype(np.uint16)
    d = np.empty((o, rb, 5), dtype=np.uint8)
    for k in range(5):
        d[:, :, k] = (x // (3 ** k)) % 3
    return d.reshape(o, rb * 5)[:, :i]


def _five_value(blob: bytes, shape, raw: dict | None = None) -> np.ndarray:
    o, i = shape
    rb = (i + 4) // 5
    codes = _trits(blob[: o * rb], o, i)
    nzmask = codes != 1
    nnz = int(nzmask.sum())
    rbytes = (nnz + 7) // 8
    off = o * rb
    bits = np.unpackbits(np.frombuffer(blob[off: off + rbytes], dtype=np.uint8),
                         bitorder="little")[:nnz].astype(bool)
    off += rbytes
    lo = np.frombuffer(blob[off: off + 2 * o], dtype=np.float16)
    hi = np.frombuffer(blob[off + 2 * o: off + 4 * o], dtype=np.float16)
    assert off + 4 * o == len(blob), (off + 4 * o, len(blob))
    is_hi = np.zeros((o, i), dtype=bool)
    is_hi[nzmask] = bits
    mag = np.where(is_hi, hi[:, None], lo[:, None])
    sign = codes.astype(np.int8) - 1
    if raw is not None:
        raw.update(sign=sign, is_hi=is_hi, lo=lo.copy(), hi=hi.copy())
    return (sign.astype(np.float16) * mag).astype(np.float16)


def _intn(blob: bytes, shape, bits: int, raw: dict | None = None) -> np.ndarray:
    o = shape[0]
    total = int(np.prod(shape))
    ncol = total // o
    sc_b = 2 * o
    body, scales = blob[:-sc_b], np.frombuffer(blob[-sc_b:], dtype=np.float16)
    if bits == 8:
        q = np.frombuffer(body, dtype=np.int8).astype(np.int32)[:total]
    elif bits == 6:
        b = np.frombuffer(body, dtype=np.uint8).reshape(-1, 3).astype(np.uint32)
        packed = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        u = np.stack([(packed >> s) & 0x3F for s in (0, 6, 12, 18)], axis=1).ravel()
        q = u[:total].astype(np.int32) - 32
    else:
        raise ValueError(bits)
    if raw is not None:
        raw.update(q=q.reshape(o, ncol).astype(np.int8), scale=scales.copy(), bits=bits)
    w = q.reshape(o, ncol).astype(np.float32) * scales.astype(np.float32)[:, None]
    return w.reshape(shape)


def read_container(path: str, *, with_raw: bool = False):
    """-> (tensors, index) or, with_raw=True, (tensors, index, raw).

    tensors: dict[str, np.ndarray] HF-named; five-value keys are emitted as
    `<module>.weight` like the dense state_dict export.
    raw (with_raw): per record name -> five_value {sign int8 [O,I] in {-1,0,1},
    is_hi bool [O,I], lo fp16 [O], hi fp16 [O]} | intN {q int8 [O,cols],
    scale fp16 [O], bits}.  fp16 records are not in raw."""
    raw_all = {} if with_raw else None
    with open(path, "rb") as fh:
        n = int.from_bytes(fh.read(8), "little")
        header = json.loads(fh.read(n))
        assert header["format"] == FORMAT, header["format"]
        out = {}
        for e in header["index"]:
            blob = fh.read(e["b"])
            assert len(blob) == e["b"]
            k, shape = e["k"], tuple(e["shape"])
            r = {} if with_raw else None
            if k == "five_value":
                out[e["n"] + ".weight"] = _five_value(blob, shape, r)
            elif k.startswith("int"):
                out[e["n"]] = _intn(blob, shape, int(k[3:]), r)
            elif k == "fp16":
                out[e["n"]] = np.frombuffer(blob, dtype=np.float16).reshape(shape).copy()
            else:
                raise ValueError(k)
            if with_raw and r:
                raw_all[e["n"]] = r
        assert fh.read(1) == b"", "trailing bytes"
    if with_raw:
        return out, header["index"], raw_all
    return out, header["index"]
