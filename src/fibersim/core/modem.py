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
            # Gray mapping per axis: 00->+3, 01->+1, 10->-1, 11->-3
            lvl = xp.where(two == 0, 3.0,
                xp.where(two == 1, 1.0,
                xp.where(two == 2, -1.0, -3.0)))
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
        # Inverse Gray mapping for QPSK
        # TX: I = 1 - 2*b0  =>  b0 = (1 - I)/2  =>  b0 = 0 if I>0, else 1
        #     Q = 1 - 2*(b0 XOR b1)  =>  b0 XOR b1 = (1 - Q)/2  =>  0 if Q>0, else 1
        # Therefore: b1 = b0 XOR ((1 - Q)/2)
        b0 = (s.real < 0).astype(np.uint8)  # 0 if real>=0, 1 if real<0
        q_bit = (s.imag < 0).astype(np.uint8)  # 0 if imag>=0, 1 if imag<0
        b1 = (b0 ^ q_bit).astype(np.uint8)
        return np.column_stack([b0, b1]).ravel()
    if mod == "16QAM":
        scale = np.sqrt(10.0)
        x = np.round(np.clip(s.real * scale, -3, 3)).astype(int)
        y = np.round(np.clip(s.imag * scale, -3, 3)).astype(int)
        def lev_to_bits(v):
            # Mapeo Gray inverso por eje (debe coincidir EXACTAMENTE con gray2level del TX)
            # TX gray2level: two = 2*b1 + b0
            # 00 (two=0) -> +3
            # 01 (two=1) -> +1
            # 10 (two=2) -> -1
            # 11 (two=3) -> -3
            
            # RX inverso (symbol -> bits [b1, b0]):
            # +3 -> 00
            # +1 -> 01
            # -1 -> 10
            # -3 -> 11
            
            # Crear tabla de mapeo directo
            b1 = np.zeros_like(v, dtype=np.uint8)
            b0 = np.zeros_like(v, dtype=np.uint8)
            
            # +3 -> 00
            mask = (v == 3)
            b1[mask] = 0
            b0[mask] = 0
            
            # +1 -> 01
            mask = (v == 1)
            b1[mask] = 0
            b0[mask] = 1
            
            # -1 -> 10
            mask = (v == -1)
            b1[mask] = 1
            b0[mask] = 0
            
            # -3 -> 11
            mask = (v == -3)
            b1[mask] = 1
            b0[mask] = 1
            
            return b1, b0
        
        msb_i, lsb_i = lev_to_bits(x)
        msb_q, lsb_q = lev_to_bits(y)
        return np.column_stack([msb_i, lsb_i, msb_q, lsb_q]).ravel().astype(np.uint8)
    raise ValueError(f"Unsupported modulation: {mod}")

def ber_from_symbols(tx_syms_ref: np.ndarray, rx_syms: np.ndarray, M: int) -> float:
    """Bit error rate comparing TX reference symbols vs RX symbols.

    - Truncates to min length.
    - For BPSK, does phase alignment and thresholding on real part.
    - For QPSK/16QAM, assumes symbols are ALREADY phase-aligned (done in main.py)
      and just does hard decision + bit comparison.

    Args:
        tx_syms_ref: Reference transmitted symbols (NumPy).
        rx_syms: Received symbols (should be pre-aligned for QPSK/16QAM).
        M: Modulation order (2, 4, or 16).
    """
    mod = {2: "BPSK", 4: "QPSK", 16: "16QAM"}.get(int(M), "BPSK")

    n = min(len(tx_syms_ref), len(rx_syms))
    if n == 0:
        return float("nan")
    tx = np.asarray(tx_syms_ref[:n])
    rx = np.asarray(rx_syms[:n])

    if mod == "BPSK":
        # BPSK: align phase then threshold
        theta = phase_from_reference(rx, tx)
        rx_rot = rx * np.exp(-1j * theta)
        b_tx = (tx.real < 0).astype(np.uint8)
        b_rx = (rx_rot.real < 0).astype(np.uint8)
        return float(np.mean(b_tx ^ b_rx))

    # QPSK / 16QAM: Símbolos YA deben venir alineados desde main.py
    # Hacemos hard decision en RX (mapeo al símbolo ideal más cercano) y comparamos bits
    
    # 1. Normalizar ganancia RX (para que coincida con TX en potencia promedio)
    p_tx = np.mean(np.abs(tx) ** 2)
    p_rx = np.mean(np.abs(rx) ** 2)
    if p_rx > 1e-30:
        gain = np.sqrt(p_tx / p_rx)
        rx_normalized = rx * gain
    else:
        rx_normalized = rx
    
    # 2. Hard decision en RX: mapear cada símbolo al punto ideal más cercano
    rx_decided = slice_symbols(rx_normalized, mod)
    
    # 3. Demap ambos a bits y comparar
    b_tx = _symbols_to_bits(tx, mod)  # TX original
    b_rx = _symbols_to_bits(rx_decided, mod)  # RX después de hard decision
    
    # 4. Comparar bits
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

# ==================== NUEVAS FUNCIONES DE DEMODULACIÓN MEJORADA ====================

def temporal_sync_correlation(
    rx_syms: np.ndarray,
    tx_syms: np.ndarray,
    chunk_size: int = 4096
) -> int:
    """
    Sincronización temporal usando correlación cruzada.
    
    Args:
        rx_syms: Símbolos recibidos (downsampled)
        tx_syms: Símbolos transmitidos de referencia
        chunk_size: Tamaño del chunk para correlación (evita saturar memoria)
    
    Returns:
        lag: Desplazamiento óptimo en símbolos
    """
    from scipy.signal import correlate
    
    # Normalizar para mejorar la correlación
    tx_norm = tx_syms / (np.std(tx_syms) + 1e-20)
    rx_norm = rx_syms / (np.std(rx_syms) + 1e-20)
    
    # Usar chunk para no saturar memoria
    safe_len = min(len(tx_norm), len(rx_norm))
    chunk = min(chunk_size, safe_len)
    
    # Correlación cruzada
    corr = correlate(rx_norm[:chunk], tx_norm[:chunk], mode='full')
    lag = int(np.argmax(np.abs(corr)) - (chunk - 1))
    
    # Debug: mostrar información de correlación
    max_corr = np.max(np.abs(corr))
    print(f"[Sync] Correlación: lag={lag} símbolos, max_corr={max_corr:.3f}, chunk={chunk}")
    
    return lag

def align_signals_by_lag(
    rx_syms: np.ndarray,
    tx_syms: np.ndarray,
    lag: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Alinea las señales RX y TX según el lag calculado.
    
    Args:
        rx_syms: Símbolos recibidos
        tx_syms: Símbolos transmitidos
        lag: Desplazamiento en símbolos
    
    Returns:
        (tx_aligned, rx_aligned): Señales alineadas con mismo largo
    """
    if lag > 0:
        rx_aligned = rx_syms[lag:]
        tx_aligned = tx_syms[:len(rx_aligned)]
    else:
        tx_aligned = tx_syms[-lag:]
        rx_aligned = rx_syms[:len(tx_aligned)]
    
    # Recortar al mismo largo
    L = min(len(tx_aligned), len(rx_aligned))
    return tx_aligned[:L], rx_aligned[:L]

def carrier_phase_recovery(
    rx_syms: np.ndarray,
    tx_syms: np.ndarray,
    window_len: int = 50
) -> np.ndarray:
    """
    Recuperación de fase carrier con filtrado de media móvil.
    
    Estima y compensa el error de fase usando los datos conocidos (Data-Aided).
    El filtro de media móvil suaviza la estimación de fase, eliminando jitter.
    
    Args:
        rx_syms: Símbolos recibidos alineados (ya normalizados en potencia)
        tx_syms: Símbolos transmitidos de referencia alineados
        window_len: Longitud de la ventana del filtro de media móvil
    
    Returns:
        rx_corrected: Símbolos con fase corregida
    """
    # Estimar error de fase instantáneo: usar conj(tx)*rx
    # Esto da el error de fase que necesita SUMARSE (no restarse)
    phase_err_vec = np.conj(tx_syms) * rx_syms
    phase_inst = np.unwrap(np.angle(phase_err_vec))
    
    # Filtro de media móvil para suavizar
    if window_len > 1:
        phase_est = np.convolve(phase_inst, np.ones(window_len) / window_len, mode='same')
    else:
        phase_est = phase_inst
    
    # Corregir fase: RESTAR el error estimado (porque phase_err_vec tiene el error)
    rx_corrected = rx_syms * np.exp(-1j * phase_est)
    
    return rx_corrected

def normalize_gain(
    rx_syms: np.ndarray,
    tx_syms: np.ndarray
) -> np.ndarray:
    """
    Normaliza la ganancia de RX para que coincida con la potencia de TX.
    
    Esto es crucial para el cálculo correcto de SNR y BER, ya que
    asegura que la potencia de señal esté correctamente escalada.
    
    Args:
        rx_syms: Símbolos recibidos con fase corregida
        tx_syms: Símbolos transmitidos de referencia
    
    Returns:
        rx_normalized: Símbolos con ganancia normalizada
    """
    p_tx = np.mean(np.abs(tx_syms) ** 2)
    p_rx = np.mean(np.abs(rx_syms) ** 2)
    
    if p_rx <= 0:
        return rx_syms
    
    gain = np.sqrt(p_tx / p_rx)
    return rx_syms * gain

def calculate_snr_from_evm(
    rx_syms: np.ndarray,
    tx_syms: np.ndarray,
    edge_cut: int = 100
) -> tuple[float, float]:
    """
    Calcula SNR basado en el vector de error (EVM).
    
    Ignora los bordes para evitar transitorios de filtros.
    
    Args:
        rx_syms: Símbolos recibidos procesados (fase corregida y ganancia normalizada)
        tx_syms: Símbolos transmitidos de referencia
        edge_cut: Número de símbolos a ignorar en cada borde
    
    Returns:
        (snr_linear, snr_db): SNR en escala lineal y dB
    """
    L = len(rx_syms)
    
    # Ajustar edge_cut si la señal es muy corta
    if L < 2 * edge_cut:
        edge_cut = 0
    
    # Ignorar bordes
    if edge_cut > 0:
        rx = rx_syms[edge_cut:-edge_cut]
        tx = tx_syms[edge_cut:-edge_cut]
    else:
        rx = rx_syms
        tx = tx_syms
    
    # Calcular potencias
    error_vec = rx - tx
    p_signal = np.mean(np.abs(tx) ** 2)
    p_noise = np.mean(np.abs(error_vec) ** 2)
    
    if p_noise <= 0 or p_signal <= 0:
        return float('nan'), float('nan')
    
    snr_lin = p_signal / p_noise
    snr_db = 10.0 * np.log10(snr_lin)
    
    return float(snr_lin), float(snr_db)

def demodulate_hard_decision(
    syms: np.ndarray,
    M: int
) -> np.ndarray:
    """
    Decisión dura (Hard Decision) para cálculo de BER.
    
    Mapea símbolos continuos a los puntos más cercanos de la constelación.
    
    Args:
        syms: Símbolos a demodular
        M: Orden de modulación (2=BPSK, 4=QPSK, 16=16QAM)
    
    Returns:
        Símbolos o bits demodulados (formato depende de M)
    """
    if M == 2:  # BPSK
        # Retorna bits directamente (0 o 1)
        return (syms.real > 0).astype(int)
    
    elif M == 4:  # QPSK
        # Retorna como complejo para comparación fácil
        I_bits = (syms.real > 0).astype(int)
        Q_bits = (syms.imag > 0).astype(int)
        return I_bits + 1j * Q_bits
    
    elif M == 16:  # 16-QAM
        # Normalizar a niveles -3, -1, 1, 3
        # Potencia promedio teórica de 16QAM es 10
        p_avg = np.mean(np.abs(syms) ** 2)
        s = syms * np.sqrt(10.0 / (p_avg + 1e-20))
        
        # Decisor multinivel por eje
        def decision_axis(x):
            d = 2.0 * np.floor((x + 4.0) / 2.0) - 3.0
            return np.clip(d, -3.0, 3.0)
        
        I_dec = decision_axis(s.real)
        Q_dec = decision_axis(s.imag)
        return I_dec + 1j * Q_dec
    
    else:
        raise ValueError(f"M={M} no soportado")

def calculate_ber_improved(
    rx_syms: np.ndarray,
    tx_syms: np.ndarray,
    M: int,
    edge_cut: int = 100
) -> float:
    """
    Calcula BER usando decisiones duras con el método mejorado.
    
    Args:
        rx_syms: Símbolos recibidos procesados
        tx_syms: Símbolos transmitidos de referencia
        M: Orden de modulación
        edge_cut: Símbolos a ignorar en los bordes
    
    Returns:
        BER: Bit Error Rate
    """
    L = len(rx_syms)
    
    # Ajustar edge_cut
    if L < 2 * edge_cut:
        edge_cut = 0
    
    # Ignorar bordes
    if edge_cut > 0:
        rx = rx_syms[edge_cut:-edge_cut]
        tx = tx_syms[edge_cut:-edge_cut]
    else:
        rx = rx_syms
        tx = tx_syms
    
    # Demodular con decisión dura
    rx_dec = demodulate_hard_decision(rx, M)
    tx_dec = demodulate_hard_decision(tx, M)
    
    # Calcular BER según modulación
    if M == 2:  # BPSK
        bit_errors = np.sum(rx_dec != tx_dec)
        total_bits = len(rx_dec)
        ber = bit_errors / total_bits if total_bits > 0 else 0.0
    
    elif M == 4:  # QPSK
        # Parte Real e Imaginaria cuentan como bits distintos
        # Convertir a int para evitar problemas de comparación con floats
        rx_I = rx_dec.real.astype(int)
        rx_Q = rx_dec.imag.astype(int)
        tx_I = tx_dec.real.astype(int)
        tx_Q = tx_dec.imag.astype(int)
        
        bit_errors_I = np.sum(rx_I != tx_I)
        bit_errors_Q = np.sum(rx_Q != tx_Q)
        bit_errors = bit_errors_I + bit_errors_Q
        total_bits = 2 * len(rx_dec)
        ber = bit_errors / total_bits if total_bits > 0 else 0.0
    
    elif M == 16:  # 16-QAM
        # SER (Symbol Error Rate)
        sym_errors = np.sum(rx_dec != tx_dec)
        total_syms = len(rx_dec)
        ser = sym_errors / total_syms if total_syms > 0 else 0.0
        # BER ~ SER / log2(M) para Gray coding
        ber = ser / 4.0
    
    else:
        return float('nan')
    
    return float(ber)

def improved_demodulation_pipeline(
    rx_wave: np.ndarray,
    tx_syms: np.ndarray,
    sps: int,
    M: int,
    guess_delay: int = 0,
    edge_cut: int = 100,
    cpr_window: int = 50,
    use_correlation_sync: bool = True
) -> dict:
    """
    Pipeline completo de demodulación mejorado.
    
    Implementa el método robusto de procesamiento:
    1. Downsampling con búsqueda de fase óptima
    2. Sincronización temporal (correlación o downsampling directo)
    3. Alineación de señales
    4. Recuperación de fase carrier (CPR)
    5. Normalización de ganancia
    6. Cálculo de SNR (basado en EVM)
    7. Cálculo de BER (decisiones duras)
    
    Args:
        rx_wave: Forma de onda recibida (sobremuestreada)
        tx_syms: Símbolos transmitidos de referencia
        sps: Muestras por símbolo
        M: Orden de modulación (2, 4, 16)
        guess_delay: Estimación inicial del retardo en muestras
        edge_cut: Símbolos a ignorar en bordes para métricas
        cpr_window: Ventana del filtro CPR
        use_correlation_sync: Si True, usa correlación; si False, usa downsampling directo
    
    Returns:
        dict con: 'rx_syms', 'snr_db', 'ber', 'delay_samp', 'tx_aligned', 'rx_aligned'
    """
    
    # 1. TX: símbolos directos (generación digital, asumimos fase 0)
    tx_syms_down = tx_syms
    
    # 2. RX: Downsampling - buscar la fase óptima usando CORRELACIÓN, no potencia
    # El método de "máxima potencia" es INCORRECTO - puede elegir ruido
    # En su lugar, probamos cada fase y vemos cuál da mejor correlación con TX
    print(f"[Demod] Buscando mejor fase de muestreo entre {sps} opciones...")
    
    best_phase = 0
    best_corr = -np.inf
    
    for phase in range(sps):
        rx_test = rx_wave[phase::sps]
        L_test = min(len(tx_syms_down), len(rx_test), 1000)  # Usar primeros 1000 símbolos
        if L_test < 100:
            continue
        # Correlación normalizada
        corr_val = np.abs(np.vdot(tx_syms_down[:L_test], rx_test[:L_test]))
        if corr_val > best_corr:
            best_corr = corr_val
            best_phase = phase
    
    rx_syms_down = rx_wave[best_phase::sps]
    print(f"[Demod] Fase óptima: {best_phase}/{sps} (corr={best_corr:.2e})")
    
    # 3. Sincronización temporal
    if use_correlation_sync:
        # Método de correlación cruzada
        lag = temporal_sync_correlation(rx_syms_down, tx_syms_down, chunk_size=4096)
        tx_aligned, rx_aligned = align_signals_by_lag(tx_syms_down, rx_syms_down, lag)
        delay_used = best_phase + lag * sps
    else:
        # Método de downsampling directo con guess_delay
        # Aplicar guess_delay para compensar retardo de grupo del filtro
        delay_in_syms = guess_delay // sps
        if delay_in_syms > 0:
            rx_syms_down = rx_syms_down[delay_in_syms:]
        
        L = min(len(tx_syms_down), len(rx_syms_down))
        tx_aligned = tx_syms_down[:L]
        rx_aligned = rx_syms_down[:L]
        delay_used = guess_delay
    
    # 4. Normalización de ganancia (ANTES de CPR para que funcione correctamente)
    rx_normalized = normalize_gain(rx_aligned, tx_aligned)
    print(f"[Demod] Símbolos alineados: TX={len(tx_aligned)}, RX={len(rx_aligned)}")
    
    # 5. Recuperación de fase carrier (CPR) - ahora con potencias normalizadas
    rx_cpr = carrier_phase_recovery(rx_normalized, tx_aligned, window_len=cpr_window)
    
    # 6. Cálculo de SNR
    snr_lin, snr_db = calculate_snr_from_evm(rx_cpr, tx_aligned, edge_cut=edge_cut)
    
    # 7. Cálculo de BER
    ber = calculate_ber_improved(rx_cpr, tx_aligned, M=M, edge_cut=edge_cut)
    
    # Debug: verificar alineación y métricas
    if len(tx_aligned) > 0:
        # Verificar fase residual
        phase_check = np.angle(np.vdot(tx_aligned[:min(100, len(tx_aligned))], 
                                        rx_cpr[:min(100, len(rx_cpr))]))
        
        # Mostrar algunos símbolos para verificación
        print(f"[Demod Debug] Fase residual={np.degrees(phase_check):.1f}°")
        print(f"[Demod Debug] Pow(RX)={np.mean(np.abs(rx_cpr)**2):.6f}, Pow(TX)={np.mean(np.abs(tx_aligned)**2):.6f}")
        print(f"[Demod Debug] Primeros 5 TX: {tx_aligned[:5]}")
        print(f"[Demod Debug] Primeros 5 RX: {rx_cpr[:5]}")
        print(f"[Demod Debug] SNR={snr_db:.2f} dB, BER={ber:.6e}")
    
    return {
        'rx_syms': rx_cpr,
        'snr_linear': snr_lin,
        'snr_db': snr_db,
        'ber': ber,
        'delay_samp': delay_used,
        'tx_aligned': tx_aligned,
        'rx_aligned': rx_cpr
    }
