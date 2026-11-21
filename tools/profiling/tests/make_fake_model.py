#!/usr/bin/env python3
"""Cria um GGUF sintetico de 2 shards para testar o parser e o fluxo (sem GPU)."""
import os
import struct

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fakemodel")


def kv_string(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def write_shard(path, tensors, pad_bytes):
    # tensors: list of (name, dims, type, offset)
    h = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", len(tensors)) + struct.pack("<Q", 0)
    for name, dims, ttype, offset in tensors:
        h += kv_string(name) + struct.pack("<I", len(dims))
        for d in dims:
            h += struct.pack("<Q", d)
        h += struct.pack("<I", ttype) + struct.pack("<Q", offset)
    header_end = len(h)
    alignment = 32
    data_start = (header_end + alignment - 1) // alignment * alignment
    data = b"\0" * (pad_bytes)
    with open(path, "wb") as f:
        f.write(h)
        f.write(b"\0" * (data_start - header_end))
        f.write(data)


def main():
    os.makedirs(OUT, exist_ok=True)
    s1 = os.path.join(OUT, "fake-00001-of-00002.gguf")
    s2 = os.path.join(OUT, "fake-00002-of-00002.gguf")
    # 64*2*84/64 = 168 bytes de dados
    write_shard(s1, [("blk.0.ffn_down_exps.weight", [64, 2], 42, 0)], 168)
    # 160*100/32*18 = 9000 bytes
    write_shard(s2, [("per_layer_token_embd.weight", [160, 100], 20, 0)], 9000 * 4)
    print(s1, os.path.getsize(s1))
    print(s2, os.path.getsize(s2))


if __name__ == "__main__":
    main()
