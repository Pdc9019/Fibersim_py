from __future__ import annotations
import math
from typing import Any, Dict, Tuple
from .array_api import xp

def prbs_gen(Nsym: int, M: int, info_in: Dict[str, Any] | None) -> Tuple["xp.ndarray", Dict[str, Any]]:
    k = int(math.log2(M))
    assert 2 ** k == M, "M debe ser potencia de 2"
    nbits = Nsym * k
    bits = xp.random.randint(0, 2, size=nbits, dtype=xp.uint8)
    info = dict(info_in or {})
    info["Nsym"] = Nsym
    info["modOrder"] = M
    return bits, info
