#!/usr/bin/env python3
"""
Demo script para mostrar la inconsistencia SNR vs BER y la corrección.
"""

import math
from math import erfc, sqrt

def demonstrate_snr_ber_inconsistency():
    """Muestra la inconsistencia entre SNR teórico (OSNR) y SNR medido."""
    
    print("=== DEMOSTRACIÓN: Inconsistencia SNR vs BER ===\n")
    
    # Parámetros típicos de una simulación
    OSNR_dB = 22.0  # OSNR bueno
    Bo_Hz = 12.5e9  # Ancho de banda óptico (0.1 nm @ 1550 nm)
    Rb = 32e9       # Symbol rate típico
    
    # SNR medido real (post-DSP, incluye todos los efectos)
    SNR_measured_dB = 8.5  # Típicamente mucho menor por penalty
    
    print(f"📊 PARÁMETROS DE ENTRADA:")
    print(f"   OSNR (solo ASE): {OSNR_dB:.1f} dB")
    print(f"   SNR medido (post-DSP): {SNR_measured_dB:.1f} dB")
    print(f"   Ancho banda óptico: {Bo_Hz/1e9:.1f} GHz")
    print(f"   Symbol rate: {Rb/1e9:.1f} GBaud")
    print()
    
    # CÁLCULO TEÓRICO (método actual del simulador)
    print("🔬 CÁLCULO TEÓRICO (solo ruido ASE):")
    OSNR_lin = 10.0**(OSNR_dB/10.0)
    SNR_theoretical_lin = OSNR_lin * (Bo_Hz / Rb)
    SNR_theoretical_dB = 10.0 * math.log10(SNR_theoretical_lin)
    
    BER_theoretical = 0.5 * erfc(sqrt(SNR_theoretical_lin)/sqrt(2.0))
    
    print(f"   SNR teórico: {SNR_theoretical_dB:.1f} dB")
    print(f"   BER teórico: {BER_theoretical:.2e}")
    print()
    
    # CÁLCULO REAL (usando SNR medido)
    print("🎯 CÁLCULO REAL (SNR post-DSP):")
    SNR_measured_lin = 10.0**(SNR_measured_dB/10.0)
    BER_real = 0.5 * erfc(sqrt(SNR_measured_lin)/sqrt(2.0))
    
    print(f"   SNR medido: {SNR_measured_dB:.1f} dB")
    print(f"   BER real: {BER_real:.2e}")
    print()
    
    # ANÁLISIS DEL PENALTY
    penalty_dB = SNR_theoretical_dB - SNR_measured_dB
    ber_ratio = BER_real / BER_theoretical
    
    print("⚠️  ANÁLISIS DEL PENALTY:")
    print(f"   Penalty total: {penalty_dB:.1f} dB")
    print(f"   Factor BER: {ber_ratio:.0f}x peor")
    print()
    
    # FUENTES DEL PENALTY
    print("🔍 POSIBLES FUENTES DEL PENALTY:")
    
    # Penalty por efectos no lineales (estimación)
    fiber_length_km = 80
    power_dBm = 0  # Potencia de entrada típica
    gamma = 1.3e-3  # Coeficiente no lineal típico
    
    # Estimación burda de penalty no lineal
    nonlinear_penalty_dB = min(gamma * (10**(power_dBm/10)) * fiber_length_km * 50, 8.0)
    
    # Penalty por dispersión (estimación)
    beta2 = -21e-27  # Dispersión típica SSMF
    dispersion_penalty_dB = min(abs(beta2) * (Rb**2) * (fiber_length_km*1e3) * 1e20, 5.0)
    
    # Penalty por DSP/timing
    dsp_penalty_dB = 1.5  # Típico
    
    print(f"   - Efectos no lineales: ~{nonlinear_penalty_dB:.1f} dB")
    print(f"   - Dispersión cromática: ~{dispersion_penalty_dB:.1f} dB") 
    print(f"   - DSP/timing/phase noise: ~{dsp_penalty_dB:.1f} dB")
    
    estimated_penalty = nonlinear_penalty_dB + dispersion_penalty_dB + dsp_penalty_dB
    print(f"   - TOTAL estimado: ~{estimated_penalty:.1f} dB")
    print(f"   - MEDIDO real: {penalty_dB:.1f} dB")
    print()
    
    # RECOMENDACIONES
    print("✅ RECOMENDACIONES PARA CORREGIR:")
    print("   1. Usar SNR medido (post-DSP) para cálculo de BER")
    print("   2. Mostrar penalty total en la GUI")
    print("   3. Reportar BER teórico vs BER real")
    print("   4. Añadir modelos de penalty para estimación")
    print()
    
    # EJEMPLO DE SALIDA CORREGIDA
    print("📱 GUI CORREGIDA MOSTRARÍA:")
    print(f"   SNR símbolo: {SNR_measured_dB:.1f} dB")
    print(f"   OSNR final: {OSNR_dB:.1f} dB")
    print(f"   BER (teórico): {BER_theoretical:.2e}")
    print(f"   BER (real): {BER_real:.2e}")
    print(f"   Penalty: {penalty_dB:.1f} dB")

def compare_different_scenarios():
    """Compara diferentes escenarios para mostrar cuándo el penalty es mayor."""
    
    print("\\n=== COMPARACIÓN DE ESCENARIOS ===\\n")
    
    scenarios = [
        {"name": "Enlace corto, baja potencia", "osnr": 25, "snr_real": 22, "desc": "Dominado por ruido ASE"},
        {"name": "Enlace medio, potencia nominal", "osnr": 20, "snr_real": 12, "desc": "Mixto: ASE + nonlinear"},
        {"name": "Enlace largo, alta potencia", "osnr": 18, "snr_real": 6, "desc": "Dominado por efectos NL"},
        {"name": "Sistema ideal (sin fibra)", "osnr": 30, "snr_real": 29, "desc": "Solo limitado por ASE"}
    ]
    
    for scenario in scenarios:
        osnr = scenario["osnr"]
        snr_real = scenario["snr_real"]
        penalty = osnr - 4.1 - snr_real  # -4.1 dB es conversión Bo/Rb típica
        
        # BER teórico vs real
        osnr_lin = 10**(osnr/10)
        snr_theo_lin = osnr_lin * (12.5e9/32e9)
        ber_theo = 0.5 * erfc(sqrt(snr_theo_lin)/sqrt(2))
        
        snr_real_lin = 10**(snr_real/10)
        ber_real = 0.5 * erfc(sqrt(snr_real_lin)/sqrt(2))
        
        print(f"🔸 {scenario['name']}:")
        print(f"   OSNR: {osnr} dB → BER teórico: {ber_theo:.1e}")
        print(f"   SNR real: {snr_real} dB → BER real: {ber_real:.1e}")
        print(f"   Penalty: {penalty:.1f} dB | {scenario['desc']}")
        print()

if __name__ == "__main__":
    demonstrate_snr_ber_inconsistency()
    compare_different_scenarios()
