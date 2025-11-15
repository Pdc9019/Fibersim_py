# Patch para corregir la inconsistencia SNR vs BER
# Este archivo contiene las funciones corregidas para usar SNR medido real

def calculate_ber_from_measured_snr(snr_sym_dB, modulation_order=2):
    """
    Calcula BER usando el SNR medido real post-DSP en lugar del OSNR teórico.
    
    Args:
        snr_sym_dB: SNR medido a nivel de símbolo en dB
        modulation_order: Orden de modulación (2=BPSK, 4=QPSK, 16=16QAM)
    
    Returns:
        BER estimado basado en SNR real
    """
    import math
    from math import erfc, sqrt
    
    if snr_sym_dB is None or not math.isfinite(snr_sym_dB):
        return None
    
    SNR_lin = 10.0**(snr_sym_dB/10.0)
    
    if modulation_order == 2:  # BPSK
        # BER = 0.5 * erfc(sqrt(SNR)/sqrt(2))
        BER = 0.5 * erfc(sqrt(max(SNR_lin, 1e-12))/sqrt(2.0))
    elif modulation_order == 4:  # QPSK
        # BER ≈ 0.5 * erfc(sqrt(SNR/2))
        BER = 0.5 * erfc(sqrt(0.5*max(SNR_lin, 1e-12)))
    elif modulation_order == 16:  # 16QAM
        # BER ≈ 0.75/4 * erfc(sqrt(0.1*SNR)) (aproximación)
        BER = (0.75/4.0) * erfc(sqrt(0.1*max(SNR_lin, 1e-12)))
    else:
        # Genérico para órdenes altos
        BER = 0.2 * erfc(sqrt(0.1*max(SNR_lin, 1e-12)))
    
    return float(BER)

def add_snr_penalty_analysis(prof, snr_measured_dB=None):
    """
    Añade análisis de penalty entre OSNR teórico y SNR medido.
    
    Args:
        prof: Profile con OSNR_dB calculados
        snr_measured_dB: SNR real medido post-DSP
    """
    import math
    
    for pt in prof:
        osnr_dB = pt.get("OSNR_dB", None)
        
        # Penalty analysis
        if osnr_dB is not None and snr_measured_dB is not None:
            # Conversión OSNR → SNR teórico (12.5 GHz → 32 GBaud típico)
            Bo_Hz = 12.5e9  # Reference optical bandwidth
            Rb = 32e9       # Symbol rate (should be from config)
            
            OSNR_lin = 10.0**(osnr_dB/10.0)
            SNR_theoretical_lin = OSNR_lin * (Bo_Hz / Rb)
            SNR_theoretical_dB = 10.0 * math.log10(max(SNR_theoretical_lin, 1e-12))
            
            # Penalty total
            penalty_dB = SNR_theoretical_dB - snr_measured_dB
            
            pt["SNR_theoretical_dB"] = SNR_theoretical_dB
            pt["SNR_measured_dB"] = snr_measured_dB
            pt["penalty_total_dB"] = penalty_dB
            
            # BER con SNR real (más realista)
            pt["BER_measured"] = calculate_ber_from_measured_snr(snr_measured_dB, 2)
            
        else:
            pt["SNR_theoretical_dB"] = None
            pt["SNR_measured_dB"] = snr_measured_dB
            pt["penalty_total_dB"] = None
            pt["BER_measured"] = None

def enhanced_ber_calculation(osnr_dB, snr_measured_dB=None, modulation_order=2):
    """
    Cálculo mejorado de BER que prioriza SNR medido sobre OSNR teórico.
    
    Returns:
        dict con BER_theoretical, BER_measured, y penalty_dB
    """
    import math
    from math import erfc, sqrt
    
    result = {
        "BER_theoretical": None,
        "BER_measured": None,
        "penalty_dB": None
    }
    
    # BER teórico desde OSNR (método original)
    if osnr_dB is not None:
        Bo_Hz = 12.5e9
        Rb = 32e9  # Debería venir de configuración
        
        OSNR_lin = 10.0**(osnr_dB/10.0)
        SNR_theoretical_lin = OSNR_lin * (Bo_Hz / Rb)
        result["BER_theoretical"] = calculate_ber_from_measured_snr(
            10.0 * math.log10(max(SNR_theoretical_lin, 1e-12)), 
            modulation_order
        )
    
    # BER medido desde SNR real
    if snr_measured_dB is not None:
        result["BER_measured"] = calculate_ber_from_measured_snr(snr_measured_dB, modulation_order)
        
        # Calcular penalty si tenemos ambos
        if osnr_dB is not None:
            OSNR_lin = 10.0**(osnr_dB/10.0)
            SNR_theoretical_lin = OSNR_lin * (12.5e9 / 32e9)
            SNR_theoretical_dB = 10.0 * math.log10(max(SNR_theoretical_lin, 1e-12))
            result["penalty_dB"] = SNR_theoretical_dB - snr_measured_dB
    
    return result
