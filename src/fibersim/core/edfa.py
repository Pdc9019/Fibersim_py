from __future__ import annotations
from typing import Any, Dict, Tuple
from . import array_api as ap
from .utils import fill_defaults, getfield_def

def edfa_block(Ain, info_in: Dict[str, Any] | None, par: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    xp = ap.xp
    
    # Asegurar que Ain esté en el backend correcto
    A = ap.to_backend(Ain)
    
    par = fill_defaults(par, {"G_dB": 20.0, "nsp": 1.5})
    Rs = getfield_def(par, "Rs", getfield_def(info_in or {}, "Rb", 32e9))

    G = 10.0 ** (par["G_dB"] / 10.0)
    A = xp.sqrt(xp.array(G)) * A  # Asegurar que G esté en el backend correcto

    # ASE (modelo simple, igual a MATLAB)
    h = 6.62607015e-34
    lam = 1550e-9
    nu = 3e8 / lam
    # Potencia de ASE por polarización (factor 1/2 para una polarización)
    Pase = par["nsp"] * h * nu * (G - 1.0) * Rs / 2.0
    noise = xp.sqrt(Pase / 2.0) * (xp.random.standard_normal(A.shape) + 1j * xp.random.standard_normal(A.shape))
    Aout = A + noise

    info = dict(info_in or {})
    info["G_dB"] = getfield_def(info_in or {}, "G_dB", 0.0) + par["G_dB"]
    info["P_ase_W"] = float(Pase)  # Potencia de ASE añadida en este EDFA
    
    # Acumular ruido ASE total para cálculo de OSNR
    P_ASE_total_prev = getfield_def(info_in or {}, "P_ASE_total", 0.0)
    # El ruido anterior también se amplifica
    P_ASE_total = P_ASE_total_prev * G + Pase
    info["P_ASE_total"] = float(P_ASE_total)
    
    return Aout, info
