from __future__ import annotations
from typing import Tuple, Any
import numpy as np

# --- helpers CPU/GPU bridging ---
def _to_np(a):
    # asegura np.ndarray
    if hasattr(a, "get"):
        import numpy as _np
        return a.get()
    import numpy as _np
    return _np.asarray(a)

def map_bits_to_symbols(bits, M: int, xp):
    """
    Mapea bits -> símbolos complejos normalizados a potencia media unitaria.
    Gray por eje. Devuelve xp.ndarray complejo (complex128).
    bits: xp.ndarray de {0,1}, longitud múltiplo de log2(M).
    """
    import numpy as _np
    k = int(_np.log2(M))
    bits_xp = bits
    if hasattr(bits_xp, "dtype") and getattr(bits_xp.dtype, "kind", "") != "u":
        try:
            bits_xp = bits_xp.astype(getattr(xp, "uint8", _np.uint8))
        except Exception:
            pass
    # Move to xp for math, but we’ll reshape using xp ops
    total = int(bits_xp.size)
    assert total % k == 0, "bits length must be multiple of log2(M)"
    # Cast to signed ints to avoid uint8 wrap-around in arithmetic
    b = bits_xp.reshape(-1, k).astype(getattr(xp, "int32", int))

    if M == 2:
        s = (1.0 - 2.0 * b[:, 0].astype(xp.float64))
        return s.astype(xp.complex128)

    if M == 4:
        # Gray mapping by quadrant: Q depends on XOR(b0, b1)
        # bits per symbol: [b0, b1]
        I = (1.0 - 2.0 * b[:, 0].astype(xp.float64))
        # XOR on ints, then cast to float
        Qbits = (b[:, 0] ^ b[:, 1])
        Q = (1.0 - 2.0 * Qbits.astype(xp.float64))
        s = (I + 1j * Q) / xp.sqrt(2.0)
        return s.astype(xp.complex128)

    if M == 16:
        def gray2level(b1, b0, xp):
            two = 2 * b1 + b0
            lvl = xp.where(two == 0, 3.0,
                  xp.where(two == 1, 1.0,
                  xp.where(two == 3, -1.0, -3.0)))
            return lvl
        I = gray2level(b[:, 0], b[:, 1], xp).astype(xp.float64)
        Q = gray2level(b[:, 2], b[:, 3], xp).astype(xp.float64)
        s = (I + 1j * Q) / xp.sqrt(10.0)
        return s.astype(xp.complex128)

    raise ValueError(f"M={M} not supported")

def _map_bits_gray_numpy(bits_np: np.ndarray, M: int) -> np.ndarray:
    """Gray mapping helper using NumPy (M in {2,4,16})."""
    M = int(M)
    b = np.asarray(bits_np, dtype=np.uint8).ravel()
    if M == 2:
        return (1 - 2 * b.astype(np.int8)).astype(np.complex128)
    if M == 4:
        n_full = (b.size // 2) * 2
        b = b[:n_full].reshape(-1, 2)
        i = 1 - 2 * (b[:, 0])
        q = 1 - 2 * (b[:, 0] ^ b[:, 1])
        return ((i + 1j * q) / np.sqrt(2.0)).astype(np.complex128)
    if M == 16:
        n_full = (b.size // 4) * 4
        b = b[:n_full].reshape(-1, 4)
        def bits_to_level(msb, lsb):
            return (1 - 2 * msb) * (2 + (1 - 2 * (msb ^ lsb)))
        i_lev = bits_to_level(b[:, 0], b[:, 1])
        q_lev = bits_to_level(b[:, 2], b[:, 3])
        return ((i_lev + 1j * q_lev) / np.sqrt(10.0)).astype(np.complex128)
    raise ValueError(f"Unsupported M={M}")

def map_bits_to_symbols_xp(bits: Any, M: int, xp) -> Any:
    """Public API variant: input bits on xp backend, return xp array symbols.

    Follows Gray mapping with unit average power.
    """
    try:
        bits_np = bits if isinstance(bits, np.ndarray) else np.asarray(getattr(xp, "asnumpy", lambda a: a)(bits))
    except Exception:
        bits_np = np.asarray(bits)
    syms_np = _map_bits_gray_numpy(bits_np, int(M))
    try:
        return xp.asarray(syms_np, dtype=xp.complex128)
    except Exception:
        return syms_np

def normalize_constellation(s: Any, xp_or_none=None) -> Any:
    """Remove mean and scale to unit RMS (NumPy default; optional xp backend)."""
    if xp_or_none is None:
        x = np.asarray(s)
        if x.size == 0:
            return x
        x = x - np.mean(x)
        rms = float(np.sqrt(np.mean(np.abs(x) ** 2)))
        return x / (rms if rms > 0 else 1.0)
    try:
        x = s
        x = x - xp_or_none.mean(x)
        rms = xp_or_none.sqrt(xp_or_none.mean(xp_or_none.abs(x) ** 2))
        rf = float(rms) if hasattr(rms, "__float__") else 1.0
        return x / (rms if rf > 0 else 1.0)
    except Exception:
        return normalize_constellation(s, None)

def carrier_phase_align(r: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Align r's global phase to t using theta=angle(vdot(t,r))."""
    r = np.asarray(r).ravel(); t = np.asarray(t).ravel()
    n = min(r.size, t.size)
    if n == 0:
        return r
    theta = np.angle(np.vdot(t[:n], r[:n]))
    return r[:n] * np.exp(-1j * theta)

def slice_symbols(y: np.ndarray, mod: str) -> np.ndarray:
    """Hard decision slicer to nearest constellation point.

    Works for BPSK, QPSK, and 16QAM (unit-average power).
    """
    y = np.asarray(y)
    mod = str(mod).upper()
    if mod == "BPSK":
        return np.where(y.real >= 0, 1.0 + 0j, -1.0 + 0j)
    if mod == "QPSK":
        i = np.where(y.real >= 0, 1.0, -1.0)
        q = np.where(y.imag >= 0, 1.0, -1.0)
        return (i + 1j * q) / np.sqrt(2.0)
    if mod == "16QAM":
        # nearest level per axis in {-3,-1,1,3}/sqrt(10)
        levels = np.array([-3, -1, 1, 3], dtype=float)
        scale = np.sqrt(10.0)
        def nearest(v):
            idx = np.argmin(np.abs(levels[:, None] - v[None, :]), axis=0)
            return levels[idx]
        i = nearest(y.real * scale)
        q = nearest(y.imag * scale)
        return (i + 1j * q) / scale
    raise ValueError(f"Unsupported modulation: {mod}")

def nearest_symbol(rx: np.ndarray, M: int) -> np.ndarray:
    """Nearest symbol slicer by M (BPSK/QPSK/16QAM), NumPy backend."""
    rx = np.asarray(rx)
    if M == 2:
        return np.where(rx.real >= 0, 1.0 + 0j, -1.0 + 0j)
    if M == 4:
        i = np.where(rx.real >= 0, 1.0, -1.0)
        q = np.where(rx.imag >= 0, 1.0, -1.0)
        return (i + 1j * q) / np.sqrt(2.0)
    if M == 16:
        levels = np.array([-3, -1, 1, 3], dtype=float)
        scale = np.sqrt(10.0)
        def nearest_axis(v):
            idx = np.argmin(np.abs(levels[:, None] - (v[None, :] * scale)), axis=0)
            return levels[idx]
        i = nearest_axis(rx.real)
        q = nearest_axis(rx.imag)
        return (i + 1j * q) / scale
    raise ValueError(f"Unsupported M={M}")

def evm_db(y: np.ndarray, ref: np.ndarray) -> Tuple[float, float]:
    """Compute RMS EVM and EVM in dB against a reference symbol sequence.

    The phase is aligned (MMSE) prior to computing EVM.
    Returns (evm_rms, evm_dB).
    """
    y = np.asarray(y).ravel()
    ref = np.asarray(ref).ravel()
    n = min(y.size, ref.size)
    if n == 0:
        return float("nan"), float("nan")
    y = y[:n]; ref = ref[:n]
    # align phase
    num = np.vdot(ref, y)
    theta = np.angle(num)
    y_rot = y * np.exp(-1j * theta)
    err = y_rot - ref
    Ps = float(np.mean(np.abs(ref) ** 2))
    Pe = float(np.mean(np.abs(err) ** 2))
    if Ps <= 0:
        return float("nan"), float("nan")
    evm_rms = np.sqrt(Pe / Ps)
    evm_db_val = 20.0 * np.log10(max(evm_rms, 1e-15))
    return float(evm_rms), float(evm_db_val)

def evm_rms_db(tx_syms: np.ndarray, rx_syms: np.ndarray) -> float:
    """EVM RMS in dB against tx reference with phase alignment."""
    _rms, db = evm_db(rx_syms, tx_syms)
    return db

def q_from_evm(evm_rms: float) -> float:
    """Approximate Q-factor from EVM.

    Using Q ≈ sqrt(1/evm^2 - 1), valid when EVM relates to SNR as evm^2≈1/(1+SNR).
    """
    if not np.isfinite(evm_rms) or evm_rms <= 0:
        return float("nan")
    try:
        return float(np.sqrt(max(1.0 / (evm_rms ** 2) - 1.0, 0.0)))
    except Exception:
        return float("nan")

def q_factor_from_evm(evm_rms_linear: float) -> float:
    """Approximate Q ≈ sqrt(2)/EVM_lin."""
    try:
        if not np.isfinite(evm_rms_linear) or evm_rms_linear <= 0:
            return float("nan")
        return float(np.sqrt(2.0) / evm_rms_linear)
    except Exception:
        return float("nan")

def slice_to_symbols(x: np.ndarray, sps: int, delay_samp: int, Nsym: int | None = None) -> np.ndarray:
    """Toma símbolos cada sps empezando en delay_samp. Recorta o rellena a Nsym si se pide."""
    start = max(0, int(delay_samp))
    y = x[start::int(sps)]
    if Nsym is not None:
        if len(y) >= Nsym:
            y = y[:Nsym]
        else:
            y = np.pad(y, (0, Nsym - len(y)), mode="constant")
    return y

def phase_from_reference(rx: np.ndarray, tx_ref: np.ndarray) -> float:
    """Fase que minimiza ||rx*e^{-jθ} - tx||^2, usando <tx, rx>."""
    n = min(len(rx), len(tx_ref))
    if n == 0:
        return 0.0
    num = np.vdot(tx_ref[:n], rx[:n])  # conj(tx) @ rx
    return float(np.angle(num))

def _symbols_to_bits(s: np.ndarray, mod: str) -> np.ndarray:
    """Demap symbols back to bits using Gray mapping for QPSK/16QAM."""
    mod = str(mod).upper()
    if mod == "BPSK":
        return (s.real < 0).astype(np.uint8)
    if mod == "QPSK":
        # invert mapping used in map_bits_to_symbols (Gray)
        i = (s.real >= 0).astype(np.uint8)  # 1 if >=0 -> bit 0 for msb (0)
        q = (s.imag >= 0).astype(np.uint8)
        b0 = (1 - i).astype(np.uint8)
        b1 = (i ^ (1 - q)).astype(np.uint8)
        return np.column_stack([b0, b1]).ravel()
    if mod == "16QAM":
        scale = np.sqrt(10.0)
        x = np.round(np.clip(s.real * scale, -3, 3)).astype(int)
        y = np.round(np.clip(s.imag * scale, -3, 3)).astype(int)
        def lev_to_bits(v):
            # invert Gray per axis: +3->00, +1->01, -1->11, -3->10
            msb = (v <= 0).astype(np.uint8)
            lsb = (v == 1).astype(np.uint8) ^ msb
            return msb, lsb
        msb_i, lsb_i = lev_to_bits(x)
        msb_q, lsb_q = lev_to_bits(y)
        return np.column_stack([msb_i, lsb_i, msb_q, lsb_q]).ravel().astype(np.uint8)
    raise ValueError(f"Unsupported modulation: {mod}")

def ber_from_symbols(tx_syms_ref: np.ndarray, rx_syms: np.ndarray, M: int) -> float:
    """Bit error rate comparing TX reference symbols vs RX symbols.

    - Truncates to min length.
    - For BPSK, does phase alignment and thresholding on real part.
    - For QPSK/16QAM, hard-slice both to the ideal constellation first, then demap to bits (Gray) and compare.

    Args:
        tx_syms_ref: Reference transmitted symbols (NumPy).
        rx_syms: Received symbols.
        M: Optional M for backward compatibility (ignored if mod provided).
        mod: Modulation string.
    """
    mod = {2: "BPSK", 4: "QPSK", 16: "16QAM"}.get(int(M), "BPSK")

    n = min(len(tx_syms_ref), len(rx_syms))
    if n == 0:
        return float("nan")
    tx = np.asarray(tx_syms_ref[:n])
    rx = np.asarray(rx_syms[:n])

    if mod == "BPSK":
        theta = phase_from_reference(rx, tx)
        rx_rot = rx * np.exp(-1j * theta)
        b_tx = (tx.real < 0).astype(np.uint8)
        b_rx = (rx_rot.real < 0).astype(np.uint8)
        return float(np.mean(b_tx ^ b_rx))

    # QPSK / 16QAM: slice then compare bits
    rx_s = slice_symbols(rx, mod)
    tx_s = slice_symbols(tx, mod)
    b_rx = _symbols_to_bits(rx_s, mod)
    b_tx = _symbols_to_bits(tx_s, mod)
    m = min(b_rx.size, b_tx.size)
    if m == 0:
        return float("nan")
    return float(np.mean(b_rx[:m] ^ b_tx[:m]))

def find_best_delay(
    rx_wave: np.ndarray,
    sps: int,
    tx_syms_ref: np.ndarray,
    guess_delay: int,
    halfwin: int = 8,
) -> tuple[int, float, np.ndarray]:
    """
    Busca el retardo con BER mínimo en [guess-halfwin, guess+halfwin].
    Devuelve (best_delay, best_ber, rx_syms_best).
    """
    Nsym = len(tx_syms_ref)
    best_ber = 1.0
    best_d = guess_delay
    best_syms = None

    d0 = max(0, int(guess_delay) - int(halfwin))
    d1 = max(0, int(guess_delay) + int(halfwin))
    for d in range(d0, d1 + 1):
        s_hat = slice_to_symbols(rx_wave, sps=sps, delay_samp=d, Nsym=Nsym)
        ber = ber_from_symbols(tx_syms_ref, s_hat, M=2)
        if ber < best_ber:
            best_ber = ber
            best_d = d
            best_syms = s_hat

    if best_syms is None:
        best_syms = slice_to_symbols(rx_wave, sps=sps, delay_samp=guess_delay, Nsym=Nsym)
    return best_d, best_ber, best_syms
