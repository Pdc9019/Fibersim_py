"""
AWGN (Additive White Gaussian Noise) con corrección de ancho de banda.

Implementa ruido gaussiano blanco con escalado apropiado para sistemas
con matched filter, garantizando que el SNR especificado se alcance
a nivel de símbolo después del filtrado adaptado.

Referencias académicas:
    - Proakis & Salehi, "Digital Communications", 5th ed., Ch. 4.2
    - Barry, Lee & Messerschmitt, "Digital Communication", 3rd ed., Ch. 5.3
"""
from __future__ import annotations
from typing import Any


def add_awgn(
    signal: Any, 
    snr_db: float, 
    xp: Any,
    sps: int = 1,
    rolloff: float = 0.0,
    mode: str = "sample"
) -> Any:
    """
    Añade ruido gaussiano blanco con corrección de ancho de banda para matched filter.
    
    Para señales sobremuestreadas (sps > 1), el ruido se escala apropiadamente
    para que el SNR a nivel de símbolo (después del matched filter) sea igual a snr_db.
    
    Args:
        signal: Señal compleja en banda base (NumPy o CuPy array)
        snr_db: SNR deseado en dB
        xp: Módulo array (numpy o cupy)
        sps: Muestras por símbolo (default=1 para señales a tasa de símbolo)
        rolloff: Factor de roll-off β del RRC (default=0.0)
        mode: "sample" para SNR a nivel de muestra, "symbol" para SNR a nivel de símbolo
        
    Returns:
        Tupla (señal_con_ruido, potencia_ruido):
            - señal_con_ruido: Señal compleja con AWGN añadido
            - potencia_ruido: Potencia del ruido añadido [W]
        
    Fundamento teórico:
        En un receptor con matched filter, el ancho de banda de ruido es:
        B_n = R_s × (1 + β)
        
        donde R_s es la tasa de símbolo y β es el roll-off del RRC.
        
        Si añadimos ruido a muestras con F_s = R_s × sps:
        - Potencia de ruido en muestras: P_n,sample = N₀ × F_s
        - Potencia de ruido en símbolos: P_n,symbol = N₀ × B_n
        
        Factor de ancho de banda: F_s / B_n = sps / (1 + β)
        
        Para lograr SNR_symbol = snr_db, debemos escalar el ruido por este factor.
        
        Esto asegura que después de la integración del matched filter,
        el SNR a nivel de símbolo sea exactamente el especificado.
        
    Referencias:
        Proakis, J. G., & Salehi, M. (2008). Digital Communications (5th ed.). 
        McGraw-Hill. Sección 4.2: "Optimum Receivers for the AWGN Channel"
    """
    # Calcular potencia de la señal
    signal_power = float(xp.mean(xp.abs(signal) ** 2))
    
    # Convertir SNR de dB a lineal
    snr_linear = 10.0 ** (snr_db / 10.0)
    
    # Potencia de ruido base
    noise_power = signal_power / snr_linear
    
    # Corrección por ancho de banda del matched filter
    if mode == "symbol" and sps > 1:
        # Factor de ancho de banda de ruido
        # Después del matched filter, solo el ruido dentro de B_n = R_s(1+β) contribuye
        # Pero estamos añadiendo ruido a F_s = R_s × sps
        # Debemos escalar el ruido por el ratio de anchos de banda
        bandwidth_ratio = float(sps) / (1.0 + rolloff)
        noise_power_corrected = noise_power * bandwidth_ratio
        
        # Nota: Añadimos MÁS ruido a nivel de muestra para que después
        # del matched filter (que rechaza ruido fuera de banda), el ruido
        # dentro de banda produzca el SNR deseado a nivel de símbolo
    else:
        # mode == "sample": SNR directo a nivel de muestra (sin corrección MF)
        noise_power_corrected = noise_power
    
    # Generar ruido gaussiano complejo
    # Para ruido complejo CN(0, σ²): σ² = noise_power
    # Componentes real e imaginaria son independientes N(0, σ²/2)
    sigma = xp.sqrt(noise_power_corrected / 2.0)
    noise_real = sigma * xp.random.randn(len(signal))
    noise_imag = sigma * xp.random.randn(len(signal))
    noise = noise_real + 1j * noise_imag
    
    # Añadir ruido a la señal y retornar potencia de ruido añadida
    return signal + noise, float(noise_power_corrected)
