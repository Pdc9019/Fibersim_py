from __future__ import annotations
from typing import Any, Dict, Tuple
import math

# tomar el backend en runtime
from . import array_api as ap
from .utils import fill_defaults, getfield_def

def fiber_ssfm(
    Ain,
    info_in: Dict[str, Any],
    par: Dict[str, Any],
    dz_override: float | None = None
) -> Tuple[Any, Dict[str, Any]]:
    xp = ap.xp  # backend actual (numpy o cupy)

    par = fill_defaults(par, {"dz": 1.0, "beta2": -21e-27, "gamma": 1.3e-3, "L": 0.0, "alpha": 0.0})
    if dz_override is not None:
        par["dz"] = float(dz_override)

    Fs = float(info_in["Fs"])
    N = int(Ain.size)
    dt = 1.0 / Fs

    # frecuencias angulares
    w = 2 * xp.pi * xp.fft.fftfreq(N, d=dt)

    # pasos de SSFM
    nSteps = int(math.ceil(par["L"] / par["dz"]))
    if nSteps < 1:
        info = dict(info_in or {})
        info["Lcum"] = getfield_def(info_in, "Lcum", 0.0)
        info["beta2"] = par["beta2"]
        info["gamma"] = par["gamma"]
        # asegurar backend y calcular potencia media
        Ain_b = xp.asarray(Ain)
        info["Pmean"] = float(xp.mean(xp.abs(Ain_b) ** 2))
        return Ain, info

    dzEff = par["L"] / nSteps
    att_step = xp.exp(-par["alpha"] * dzEff / 2.0)                 # atenuación por medio paso
    Hhalf = xp.exp(-1j * 0.5 * par["beta2"] * (w ** 2) * dzEff)    # kernel lineal medio paso

    # asegurar array del backend y tipo complejo
    A = xp.asarray(Ain).astype(xp.complex128).reshape(-1)

    for _ in range(nSteps):
        # medio paso lineal
        A = xp.fft.ifft(xp.fft.fft(A) * Hhalf)
        # paso no lineal
        A = A * xp.exp(1j * par["gamma"] * xp.abs(A) ** 2 * dzEff)
        # medio paso lineal
        A = xp.fft.ifft(xp.fft.fft(A) * Hhalf)
        # atenuación
        A = A * att_step

    Aout = A.reshape(Ain.shape)

    info = dict(info_in or {})
    info["Lcum"] = getfield_def(info_in, "Lcum", 0.0) + par["L"]
    info["beta2"] = par["beta2"]
    info["gamma"] = par["gamma"]
    info["Pmean"] = float(xp.mean(xp.abs(Aout) ** 2))
    return Aout, info
