from __future__ import annotations
from typing import Any, Dict
import numpy as _np  # diseño de taps en CPU

# Importa el selector de backend en runtime
from . import array_api as ap  # ap.xp decidirá NumPy o CuPy según FIBERSIM_GPU

# Alias corto opcional
xp = ap.xp

def fill_defaults(par: Dict[str, Any] | None, defaults: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(defaults)
    if par:
        out.update(par)
    return out

def getfield_def(d: Dict[str, Any] | None, key: str, default: Any = None) -> Any:
    if not d:
        return default
    return d.get(key, default)

def _rrc_taps(beta: float, span: int, sps: int) -> _np.ndarray:
    """Raised-cosine en raíz (RRC) normalizado a ganancia unitaria a DC."""
    T = 1.0  # período símbolo normalizado
    N = span * sps
    t = _np.arange(-N/2, N/2 + 1) / sps  # en Ts
    taps = _np.zeros_like(t, dtype=_np.float64)

    for i, ti in enumerate(t):
        if abs(ti) < 1e-12:
            taps[i] = 1.0 + beta * (4/_np.pi - 1)
        elif abs(abs(4*beta*ti) - 1.0) < 1e-12:
            # singularidad en t = ±T/(4β)
            taps[i] = (beta/_np.sqrt(2)) * (
                (1+2/_np.pi) * _np.sin(_np.pi/(4*beta)) +
                (1-2/_np.pi) * _np.cos(_np.pi/(4*beta))
            )
        else:
            num = _np.sin(_np.pi*ti*(1-beta)) + 4*beta*ti*_np.cos(_np.pi*ti*(1+beta))
            den = _np.pi*ti*(1 - (4*beta*ti)**2)
            taps[i] = num / den

    # normaliza energía a 1 (ganancia ≈ 1 a DC)
    taps = taps / _np.sqrt(_np.sum(taps**2))
    return taps

def _xsignal():
    """
    Devuelve el módulo de señal adecuado según el backend activo.
    Evita mezclar SciPy con arrays de CuPy.
    """
    if getattr(ap.xp, "__name__", "") == "cupy":
        import cupyx.scipy.signal as signal
    else:
        import scipy.signal as signal
    return signal

def get_rx_filter(sps: int, roll: float, span: int):
    """Matched filter RRC: devuelve función filtro(x) consistente con el backend."""
    # Diseña taps en NumPy y súbelos al backend actual
    h = _rrc_taps(beta=roll, span=span, sps=sps).astype(_np.float64)
    h_backend = ap.xp.asarray(h)

    def filt(x):
        sig = _xsignal()  # elige SciPy o cupyx.scipy en cada llamada
        x_b = ap.xp.asarray(x)
        den = ap.xp.asarray([1.0], dtype=h_backend.dtype)
        return sig.lfilter(h_backend, den, x_b)

    return filt

def get_tx_filter(sps: int, roll: float, span: int) -> _np.ndarray:
    """Devuelve taps RRC (numpy). Se usa en TX; en GPU se suben a CuPy en el uso."""
    return _rrc_taps(beta=roll, span=span, sps=sps)
