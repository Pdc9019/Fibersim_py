from __future__ import annotations
import numpy as np

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

def ber_from_symbols(tx_syms_ref: np.ndarray, rx_syms: np.ndarray, M: int = 2) -> float:
    """BER vs referencia conocida. Por ahora BPSK."""
    n = min(len(tx_syms_ref), len(rx_syms))
    if n == 0:
        return float("nan")
    if M != 2:
        raise NotImplementedError("BER solo BPSK por ahora")

    rx = rx_syms[:n]
    tx = tx_syms_ref[:n]

    theta = phase_from_reference(rx, tx)
    rx_rot = rx * np.exp(-1j * theta)

    b_tx = (tx.real < 0).astype(np.uint8)
    b_rx = (rx_rot.real < 0).astype(np.uint8)
    return float(np.mean(b_tx ^ b_rx))

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
