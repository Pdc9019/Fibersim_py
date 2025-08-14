from __future__ import annotations
from typing import Any, Dict, Tuple
from .array_api import xp
from .utils import fill_defaults, getfield_def

def edfa_block(Ain, info_in: Dict[str, Any] | None, par: Dict[str, Any]) -> Tuple["xp.ndarray", Dict[str, Any]]:
    par = fill_defaults(par, {"G_dB": 20.0, "nsp": 1.5})
    Rs = getfield_def(par, "Rs", getfield_def(info_in or {}, "Rb", 32e9))

    G = 10.0 ** (par["G_dB"] / 10.0)
    A = xp.sqrt(G) * Ain

    # ASE (modelo simple, igual a MATLAB)
    h = 6.62607015e-34
    lam = 1550e-9
    nu = 3e8 / lam
    Pase = par["nsp"] * h * nu * (G - 1.0) * Rs
    noise = xp.sqrt(Pase / 2.0) * (xp.random.standard_normal(A.shape) + 1j * xp.random.standard_normal(A.shape))
    Aout = A + noise

    info = dict(info_in or {})
    info["G_dB"] = getfield_def(info_in or {}, "G_dB", 0.0) + par["G_dB"]
    return Aout, info
