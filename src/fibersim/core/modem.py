from __future__ import annotations
from typing import Tuple
import numpy as np

def map_bits_to_symbols(bits: np.ndarray, M: int) -> np.ndarray:
    """
    Devuelve símbolos complejos normalizados a potencia unitaria promedio.
    BPSK, QPSK Gray, 16QAM Gray.
    """
    if M == 2:
        # BPSK: {+1, -1}
        return (1 - 2*bits.astype(np.int8)).astype(np.complex128)
    k = int(np.log2(M))
    b = bits.reshape(-1, k)
    if M == 4:
        # Gray QPSK: 00 -> 1+1j, 01 -> -1+1j, 11 -> -1-1j, 10 -> 1-1j (rotaciones equivalentes)
        I = 1 - 2*b[:,0]
        Q = 1 - 2*b[:,1]
        s = (I + 1j*Q) / np.sqrt(2.0)
        return s.astype(np.complex128)
    if M == 16:
        # 16QAM Gray sobre PAM {±1, ±3} normalizado
        def pam(bit2):
            # mapea 2 bits a {+3, +1, -1, -3}
            return np.array([3,1,-1,-3], dtype=np.float64)[bit2]
        idxI = b[:,0]*2 + b[:,1]
        idxQ = b[:,2]*2 + b[:,3]
        I = pam(idxI)
        Q = pam(idxQ)
        s = (I + 1j*Q)/np.sqrt(10.0)  # potencia media 1
        return s.astype(np.complex128)
    raise ValueError(f"M no soportado: {M}")

def slicer_symbols(s_rx: np.ndarray, M: int) -> np.ndarray:
    """
    Devuelve símbolos de decisión dura en el alfabeto respectivo.
    """
    if M == 2:
        return np.where(s_rx.real >= 0, 1.0, -1.0).astype(np.complex128)
    if M == 4:
        I = np.where(s_rx.real >= 0, 1.0, -1.0)
        Q = np.where(s_rx.imag >= 0, 1.0, -1.0)
        return ((I + 1j*Q)/np.sqrt(2.0)).astype(np.complex128)
    if M == 16:
        # decisor por umbrales en ±2
        def qpam(x):
            # devuelve 3,1,-1,-3
            r = np.empty_like(x)
            r[x >= 2/np.sqrt(10.0)] = 3
            r[(x < 2/np.sqrt(10.0)) & (x >= 0)] = 1
            r[(x < 0) & (x >= -2/np.sqrt(10.0))] = -1
            r[x < -2/np.sqrt(10.0)] = -3
            return r
        I = qpam(s_rx.real)
        Q = qpam(s_rx.imag)
        return ((I + 1j*Q)/np.sqrt(10.0)).astype(np.complex128)
    raise ValueError(f"M no soportado: {M}")

def ber_from_symbols(s_tx: np.ndarray, s_hat: np.ndarray, M: int) -> float:
    """
    BER por decisión dura con Gray, estimando bits desde símbolos.
    """
    if M == 2:
        # BPSK 1 bit por símbolo
        b_tx = (s_tx.real < 0).astype(np.uint8)
        b_rx = (s_hat.real < 0).astype(np.uint8)
        return float(np.mean(b_tx ^ b_rx))
    # Para QPSK y 16QAM usa tabla inversa
    # aproximación: tasa de símbolos errados dividida por log2(M)
    k = int(np.log2(M))
    sym_err = np.mean(s_tx != s_hat)
    return float(sym_err / k)
