"""
Funciones para calcular métricas de calidad de señal (SNR, EVM, etc.)
"""

import numpy as np
from typing import Tuple

def calc_snr_post_dsp(rx_syms: np.ndarray, tx_syms: np.ndarray) -> Tuple[float, float]:
    """
    Calcula el SNR efectivo después del DSP usando una técnica mejorada que considera
    la estructura de la constelación y compensa efectos de fase y amplitud.
    
    Esta función:
    1. Detecta el tipo de modulación (BPSK, QPSK, etc.)
    2. Normaliza y alinea las constelaciones
    3. Compensa rotación de fase y offset
    4. Calcula SNR considerando las dimensiones relevantes
    
    Args:
        rx_syms: Símbolos recibidos después del DSP
        tx_syms: Símbolos transmitidos de referencia
    
    Returns:
        Tuple[float, float]: (SNR lineal, SNR en dB)
    """
    n = min(len(rx_syms), len(tx_syms))
    if n == 0:
        return float("nan"), float("nan")
    
    rx = rx_syms[:n]
    tx = tx_syms[:n]
    
    # 1. Detectar tipo de modulación
    unique_mags = np.unique(np.round(np.abs(tx), decimals=3))
    is_bpsk = len(unique_mags) == 1 and len(np.unique(np.round(np.angle(tx) % np.pi))) == 2
    is_qpsk = len(unique_mags) == 1 and len(np.unique(np.round(np.angle(tx) % (np.pi/2)))) == 4
    
    # 2. Normalizar a potencia unitaria
    rx_power = np.mean(np.abs(rx) ** 2)
    tx_power = np.mean(np.abs(tx) ** 2)
    rx_norm = rx / np.sqrt(rx_power)
    tx_norm = tx / np.sqrt(tx_power)
    
    # 3. Alinear fase usando estimación MMSE
    theta = np.angle(np.vdot(tx_norm, rx_norm))
    rx_aligned = rx_norm * np.exp(-1j * theta)
    
    if is_bpsk:
        # Para BPSK, usar solo la dimensión real
        rx_real = np.real(rx_aligned)
        tx_real = np.real(tx_norm)
        
        # Remover cualquier offset DC residual
        dc_offset = np.mean(rx_real - tx_real)
        rx_real = rx_real - dc_offset
        
        # Calcular SNR solo en dimensión real
        error_real = rx_real - tx_real
        P_signal = np.mean(tx_real ** 2)
        P_error = np.mean(error_real ** 2)
        
    elif is_qpsk:
        # Para QPSK, considerar ambas dimensiones pero compensar offsets por separado
        rx_real = np.real(rx_aligned)
        rx_imag = np.imag(rx_aligned)
        tx_real = np.real(tx_norm)
        tx_imag = np.imag(tx_norm)
        
        # Remover offsets DC por dimensión
        dc_offset_real = np.mean(rx_real - tx_real)
        dc_offset_imag = np.mean(rx_imag - tx_imag)
        rx_real = rx_real - dc_offset_real
        rx_imag = rx_imag - dc_offset_imag
        
        # Reconstruir señal compensada
        rx_comp = rx_real + 1j * rx_imag
        error = rx_comp - tx_norm
        P_signal = np.mean(np.abs(tx_norm) ** 2)
        P_error = np.mean(np.abs(error) ** 2)
        
    else:
        # Para otras modulaciones, usar método general
        error = rx_aligned - tx_norm
        P_signal = np.mean(np.abs(tx_norm) ** 2)
        P_error = np.mean(np.abs(error) ** 2)
    
    # Validar y calcular SNR
    if P_error <= 0 or P_signal <= 0:
        return float("nan"), float("nan")
    
    snr_linear = P_signal / P_error
    snr_db = 10.0 * np.log10(snr_linear)
    
    return float(snr_linear), float(snr_db)