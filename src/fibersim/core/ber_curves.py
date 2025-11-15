# src/fibersim/core/ber_curves.py
"""
Funciones para generar curvas BER teóricas vs Eb/No para diferentes modulaciones.
Basado en las ecuaciones estándar de comunicaciones digitales.
"""

import numpy as np
from math import erfc, sqrt, log10
from typing import List, Tuple

def ber_bpsk_theoretical(ebno_db: np.ndarray) -> np.ndarray:
    """
    BER teórico para BPSK en canal AWGN.
    BER = 0.5 * erfc(sqrt(Eb/No))
    """
    ebno_linear = 10.0**(ebno_db/10.0)
    ber = 0.5 * np.array([erfc(sqrt(max(eb, 1e-12))) for eb in ebno_linear])
    return ber

def ber_qpsk_theoretical(ebno_db: np.ndarray) -> np.ndarray:
    """
    BER teórico para QPSK en canal AWGN.
    BER = 0.5 * erfc(sqrt(Eb/No))
    Nota: QPSK tiene la misma curva que BPSK a nivel de bit
    """
    return ber_bpsk_theoretical(ebno_db)

def ber_16qam_theoretical(ebno_db: np.ndarray) -> np.ndarray:
    """
    BER teórico aproximado para 16-QAM en canal AWGN.
    Aproximación: BER ≈ (3/8) * erfc(sqrt(0.4 * Eb/No))
    """
    ebno_linear = 10.0**(ebno_db/10.0)
    ber = (3.0/8.0) * np.array([erfc(sqrt(0.4 * max(eb, 1e-12))) for eb in ebno_linear])
    return ber

def ber_64qam_theoretical(ebno_db: np.ndarray) -> np.ndarray:
    """
    BER teórico aproximado para 64-QAM en canal AWGN.
    Aproximación: BER ≈ (7/24) * erfc(sqrt(Eb/No / 21))
    """
    ebno_linear = 10.0**(ebno_db/10.0)
    ber = (7.0/24.0) * np.array([erfc(sqrt(max(eb, 1e-12) / 21.0)) for eb in ebno_linear])
    return ber

def generate_ber_curves(ebno_range_db: Tuple[float, float] = (0, 20), 
                       num_points: int = 200) -> dict:
    """
    Genera curvas BER teóricas para múltiples modulaciones.
    
    Args:
        ebno_range_db: Rango de Eb/No en dB (min, max)
        num_points: Número de puntos en cada curva
    
    Returns:
        dict con arrays de EbNo_dB y BER para cada modulación
    """
    ebno_db = np.linspace(ebno_range_db[0], ebno_range_db[1], num_points)
    
    curves = {
        'EbNo_dB': ebno_db,
        'BPSK': ber_bpsk_theoretical(ebno_db),
        'QPSK': ber_qpsk_theoretical(ebno_db),
        '16QAM': ber_16qam_theoretical(ebno_db),
        '64QAM': ber_64qam_theoretical(ebno_db)
    }
    
    return curves

def snr_to_ebno(snr_db: float, modulation_order: int, code_rate: float = 1.0) -> float:
    """
    Convierte SNR a Eb/No.
    
    Args:
        snr_db: SNR en dB
        modulation_order: Orden de modulación (2, 4, 16, 64)
        code_rate: Tasa de código (default 1.0 para no codificado)
    
    Returns:
        Eb/No en dB
    """
    # log2(M) bits por símbolo
    bits_per_symbol = log10(modulation_order) / log10(2)
    
    # Eb/No = SNR - 10*log10(bits_per_symbol) + 10*log10(code_rate)
    ebno_db = snr_db - 10.0 * log10(bits_per_symbol) + 10.0 * log10(code_rate)
    
    return ebno_db

def get_ber_from_ebno(ebno_db: float, modulation: str) -> float:
    """
    Obtiene BER teórico para un Eb/No específico y modulación.
    
    Args:
        ebno_db: Eb/No en dB
        modulation: 'BPSK', 'QPSK', '16QAM', '64QAM'
    
    Returns:
        BER teórico
    """
    ebno_array = np.array([ebno_db])
    
    if modulation.upper() == 'BPSK':
        return ber_bpsk_theoretical(ebno_array)[0]
    elif modulation.upper() == 'QPSK':
        return ber_qpsk_theoretical(ebno_array)[0]
    elif modulation.upper() == '16QAM':
        return ber_16qam_theoretical(ebno_array)[0]
    elif modulation.upper() == '64QAM':
        return ber_64qam_theoretical(ebno_array)[0]
    else:
        # Default to BPSK
        return ber_bpsk_theoretical(ebno_array)[0]
