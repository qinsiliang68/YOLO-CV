from __future__ import annotations
import hashlib


def derive_seed(*parts: object, bits: int = 31) -> int:
    text = "|".join(str(p) for p in parts).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(text).digest()[:8], "big")
    return value & ((1 << bits) - 1)
