from __future__ import annotations
from typing import Any, Dict, Tuple
from .array_api import xp
from .utils import fill_defaults, getfield_def

def fiber_ssfm(Ain, info_in: Dict[str, Any], par: Dict[str, Any], dz_override: float | None = None) -> Tuple["xp.ndarray", Dict[str, Any]]:
    par = fill_defaults(par, {"dz": 1.0, "beta2": -21e-27, "gamma": 1.3e-3, "L": 0.0, "alpha": 0.0})
    if dz_override is not None:
        par["dz"] = float(dz_override)

    Fs = info_in["Fs"]
    N = Ain.size
    dt = 1.0 / Fs
    w = 2 * xp.pi * xp.fft.fftfreq(N, d=dt)  # rad/s

    nSteps = int(xp.ceil(par["L"] / par["dz"]))
    if nSteps < 1:
        # tramos de L=0
        info = dict(info_in or {})
        info["Lcum"] = getfield_def(info_in, "Lcum", 0.0)
        info["beta2"] = par["beta2"]
        info["gamma"] = par["gamma"]
        info["Pmean"] = float(xp.mean(xp.abs(Ain) ** 2))
        return Ain, info

    dzEff = par["L"] / nSteps
    att_step = xp.exp(-par["alpha"] * dzEff / 2.0)   # amplitud
    Hhalf = xp.exp(-1j * 0.5 * par["beta2"] * (w ** 2) * dzEff)

    A = Ain.astype(xp.complex128).reshape(-1)
    for _ in range(nSteps):
        A = xp.fft.ifft(xp.fft.fft(A) * Hhalf)
        A = A * xp.exp(1j * par["gamma"] * xp.abs(A) ** 2 * dzEff)
        A = xp.fft.ifft(xp.fft.fft(A) * Hhalf)
        A = A * att_step

    Aout = A.reshape(Ain.shape)

    info = dict(info_in or {})
    info["Lcum"] = getfield_def(info_in, "Lcum", 0.0) + par["L"]
    info["beta2"] = par["beta2"]
    info["gamma"] = par["gamma"]
    info["Pmean"] = float(xp.mean(xp.abs(Aout) ** 2))
    return Aout, info
