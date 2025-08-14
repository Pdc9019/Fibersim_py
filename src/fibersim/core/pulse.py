from __future__ import annotations
from typing import Any, Dict, Tuple
from .array_api import xp, xsignal
from .utils import fill_defaults, get_tx_filter

def pulse_shaper(syms, info_in: Dict[str, Any] | None, par: Dict[str, Any]) -> Tuple["xp.ndarray", Dict[str, Any]]:
    par = fill_defaults(par, {"type": "RRC", "roll": 0.1, "span": 10})
    sps = round(par["Fs"] / par["Rb"])

    # Upsample (en el backend actual: NumPy o CuPy)
    up = xp.zeros((syms.size * sps,), dtype=xp.complex128)
    up[::sps] = syms

    # Taps RRC en numpy -> pásalos al backend actual con xp.asarray
    h_np = get_tx_filter(sps=sps, roll=par["roll"], span=par["span"])
    h = xp.asarray(h_np, dtype=xp.float64)

    # Filtrado en el backend actual (xsignal == cupyx.scipy.signal o scipy.signal)
    den = xp.asarray([1.0], dtype=h.dtype)   # <-- 1-D
    y = xsignal.lfilter(h, den, up)

    info = dict(info_in or {})
    info["sps"] = sps
    info["pulseDelay"] = (par["span"] * sps) // 2
    info["Fs"] = par["Fs"]
    info["Rb"] = par["Rb"]
    return y, info
