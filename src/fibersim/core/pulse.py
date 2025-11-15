from __future__ import annotations
from typing import Any, Dict, Tuple
from . import array_api as ap
from .utils import fill_defaults, get_tx_filter

def pulse_shaper(syms, info_in: Dict[str, Any] | None, par: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    xp, xsignal = ap.xp, ap.xsignal  # backend actual
    
    par = fill_defaults(par, {"type": "RRC", "roll": 0.1, "span": 10})
    sps = round(par["Fs"] / par["Rb"])

    # Upsample (en el backend actual: NumPy o CuPy)
    syms = ap.to_backend(syms)  # Asegurar backend correcto
    up = xp.zeros((syms.size * sps,), dtype=xp.complex128)
    up[::sps] = syms

    # Taps RRC en numpy -> pásalos al backend actual con xp.asarray
    h_np = get_tx_filter(sps=sps, roll=par["roll"], span=par["span"])
    h = xp.asarray(h_np, dtype=xp.float64)

    # Filtrado en el backend actual (xsignal == cupyx.scipy.signal o scipy.signal)
    den = xp.asarray([1.0], dtype=h.dtype)   # <-- 1-D
    y = xsignal.lfilter(h, den, up)
    
    # Normalizar potencia: el filtro RRC (energía=1) reduce la potencia por factor sps
    # debido al upsampling. Necesitamos escalar para que la potencia media = 1
    # para que luego Ein = sqrt(Ptx) * y produzca potencia = Ptx
    P_actual = float(xp.mean(xp.abs(y) ** 2))
    if P_actual > 0:
        y = y / xp.sqrt(P_actual)

    info = dict(info_in or {})
    info["sps"] = sps
    info["pulseDelay"] = (par["span"] * sps) // 2
    info["Fs"] = par["Fs"]
    info["Rb"] = par["Rb"]
    return y, info
