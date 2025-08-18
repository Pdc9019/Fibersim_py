from __future__ import annotations
from typing import Any, Dict, Tuple
import numpy as np
from scipy import signal as sp_signal

# We only use NumPy/SciPy here. No array_api/xp inside.

def _rrc_taps_np(beta: float, span: int, sps: int) -> np.ndarray:
    """Raised Root Cosine taps in NumPy, unit-energy normalization.

    Matches TX generator used in utils._rrc_taps.
    """
    N = span * sps
    t = np.arange(-N / 2, N / 2 + 1, dtype=np.float64) / sps
    taps = np.zeros_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-12:
            taps[i] = 1.0 + beta * (4 / np.pi - 1)
        elif abs(abs(4 * beta * ti) - 1.0) < 1e-12:
            taps[i] = (beta / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
            )
        else:
            num = np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))
            den = np.pi * ti * (1 - (4 * beta * ti) ** 2)
            taps[i] = num / den
    taps = taps / np.sqrt(np.sum(taps**2))
    return taps.astype(np.float64)

def rrc_matched_filter_np(y: np.ndarray, sps: int, roll: float, span: int) -> np.ndarray:
    """RRC matched filtering using SciPy lfilter. NumPy-only inside.

    Args:
        y: baseband waveform (NumPy array). If from CuPy, convert before calling.
        sps: samples per symbol
        roll: roll-off
        span: filter span in symbols
    """
    h = _rrc_taps_np(beta=float(roll), span=int(span), sps=int(sps))
    return sp_signal.lfilter(h, [1.0], np.asarray(y))

def timing_mm_np(y_mf: np.ndarray, sps: int, iters: int | None = None) -> Tuple[np.ndarray, int]:
    """Integer-rate timing selection by maximizing energy of decimated sequence.

    This approximates Mueller–Müller at integer phases by a simple energy proxy.

    Returns:
        y_sync: downsampled symbols at symbol rate
        delay_samp: chosen decimation phase [0..sps-1]
    """
    sps = int(sps)
    best_e = -1.0
    best_d = 0
    for d in range(sps):
        y_d = y_mf[d::sps]
        e = float(np.mean(np.abs(y_d) ** 2)) if y_d.size > 0 else -1.0
        if e > best_e:
            best_e = e
            best_d = d
    y_sync = y_mf[best_d::sps]
    return y_sync, best_d

def cma_equalizer_np(y: np.ndarray, taps: int = 11, mu: float = 1e-3, iters: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Single-input CMA equalizer (complex, SISO).

    Args:
        y: input at symbol rate (NumPy complex1d)
        taps: number of taps (odd preferred)
        mu: step size
        iters: if provided, limit to this many output samples

    Returns:
        y_eq: equalized output
        w: final coefficients (complex)
    """
    y = np.asarray(y).astype(np.complex128)
    L = int(taps)
    if L < 3: L = 3
    if L % 2 == 0: L += 1
    N = y.size
    M = N if iters is None else min(N, int(iters))
    w = np.zeros(L, dtype=np.complex128)
    w[L // 2] = 1.0 + 0j
    R2 = np.mean(np.abs(y) ** 2) if N > 0 else 1.0
    R2 = float(R2) if R2 > 0 else 1.0
    xpad = np.pad(y, (L - 1, 0), mode="constant")
    out = np.zeros(M, dtype=np.complex128)
    for n in range(M):
        xn = xpad[n:n + L][::-1]
        z = np.vdot(w, xn)
        out[n] = z
        e = (R2 - np.abs(z) ** 2) * z
        w = w + mu * e.conjugate() * xn
    return out, w

def carrier_phase_bps_np(y: np.ndarray, M: int = 4, ntest: int = 64, win: int = 64) -> np.ndarray:
    """Basic Blind Phase Search (BPS) for QPSK-like constellations.

    For 16QAM, BPS still works but performance may be limited; consider VV or DD.
    """
    y = np.asarray(y).astype(np.complex128)
    N = y.size
    if N == 0:
        return y
    thetas = np.linspace(0, 2 * np.pi / M, ntest, endpoint=False)
    out = np.zeros_like(y)
    for i in range(0, N, win):
        seg = y[i:i + win]
        if seg.size == 0:
            continue
        errs = []
        for th in thetas:
            zr = seg * np.exp(-1j * th)
            # decision-directed error proxy: distance to nearest constellation point
            if M == 4:
                i_dec = np.where(zr.real >= 0, 1.0, -1.0)
                q_dec = np.where(zr.imag >= 0, 1.0, -1.0)
                dec = (i_dec + 1j * q_dec) / np.sqrt(2.0)
            else:
                # fallback to QPSK grid for simplicity
                i_dec = np.where(zr.real >= 0, 1.0, -1.0)
                q_dec = np.where(zr.imag >= 0, 1.0, -1.0)
                dec = (i_dec + 1j * q_dec) / np.sqrt(2.0)
            errs.append(np.mean(np.abs(zr - dec) ** 2))
        k = int(np.argmin(errs))
        out[i:i + win] = seg * np.exp(-1j * thetas[k])
    return out

def carrier_phase_vv_np(y: np.ndarray, M: int = 4) -> np.ndarray:
    """Viterbi–Viterbi carrier phase estimator for M=4 (QPSK).
    Applies 4th-power method with smoothing.
    """
    y = np.asarray(y).astype(np.complex128)
    if M != 4 or y.size == 0:
        return y
    z4 = y ** 4
    # simple smoothing
    h = np.ones(33) / 33.0
    z4f = np.convolve(z4, h, mode="same")
    phi = np.angle(z4f) / 4.0
    return y * np.exp(-1j * phi)

def coherent_rx_pipeline(
    y_bb: Any, sps: int, roll: float, span: int, mod: str, dsp_par: Dict[str, Any] | None
) -> Dict[str, Any]:
    """Coherent RX pipeline (NumPy + SciPy only inside).

    Converts CuPy input to NumPy on entry; returns NumPy arrays.
    """
    # Convert CuPy to NumPy if needed
    try:
        import cupy as cp  # type: ignore
        if isinstance(y_bb, cp.ndarray):
            y = cp.asnumpy(y_bb)
        else:
            y = np.asarray(y_bb)
    except Exception:
        y = np.asarray(y_bb)

    mod = str(mod).upper()
    par = dict(dsp_par or {})
    timing_algo = str(par.get("timing_algo", "mm"))
    eq_taps = int(par.get("eq_taps", 11))
    eq_mu = float(par.get("eq_mu", 1e-3))
    phase_algo = str(par.get("phase_algo", "bps"))

    y_mf = rrc_matched_filter_np(y, int(sps), float(roll), int(span))

    if timing_algo == "mm":
        y_sync, d_sym = timing_mm_np(y_mf, int(sps))
    else:
        # default: pick center of pulse span
        d_sym = 0
        y_sync = y_mf[d_sym::int(sps)]

    y_eq, _w = cma_equalizer_np(y_sync, taps=eq_taps, mu=eq_mu)

    if phase_algo == "bps":
        y_cpr = carrier_phase_bps_np(y_eq, M=4 if mod == "QPSK" else 4)
    elif phase_algo == "vv":
        y_cpr = carrier_phase_vv_np(y_eq, M=4)
    else:
        y_cpr = y_eq

    return {
        "y_mf": y_mf,
        "y_sync": y_sync,
        "y_eq": y_eq,
        "y_cpr": y_cpr,
        "delay_sym_samp": int(d_sym),
    }
