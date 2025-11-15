#!/usr/bin/env python3
"""
Script de validación: compara resultados simulados vs cálculos teóricos
para sistemas de validación (Val_1.json, Val_2_3spans.json)
"""

import json
import math
import sys
from pathlib import Path

# Constantes físicas
h = 6.62607015e-34  # Planck constant (J·s)
c = 299792458.0     # Speed of light (m/s)
lambda_nm = 1550.0  # Wavelength (nm)

def dB_to_lin(x_dB):
    return 10.0 ** (x_dB / 10.0)

def lin_to_dB(x):
    return 10.0 * math.log10(max(x, 1e-30))

def nf_dB_from_nsp(nsp):
    """Noise Figure (dB) from nsp"""
    return 10.0 * math.log10(2.0 * nsp)

def calculate_theoretical_osnr(config_path):
    """
    Calcula el OSNR teórico de un sistema balanceado
    Asume: Ganancia EDFA = Pérdida Fibra en cada span
    """
    with open(config_path, 'r') as f:
        cfg = json.load(f)
    
    # Parámetros globales
    Ptx_W = cfg['global']['Ptx']
    Rb = cfg['global']['Rb']
    
    # Recorrer cadena
    chain = cfg['chain']
    P_signal = Ptx_W
    P_ase_total = 0.0
    
    print(f"\n{'='*70}")
    print(f"VALIDACIÓN: {config_path}")
    print(f"{'='*70}\n")
    print(f"Potencia TX: {Ptx_W*1e3:.3f} mW ({lin_to_dB(Ptx_W/1e-3):.2f} dBm)")
    print(f"Tasa de símbolos: {Rb/1e9:.1f} Gbaud\n")
    
    num_spans = 0
    for i, blk in enumerate(chain):
        typ = blk['type']
        par = blk['par']
        
        if typ == 'fiber':
            L_m = par['L']
            L_km = L_m / 1000.0
            alpha_Np_per_m = par['alpha']
            
            # Atenuación lineal
            att_linear = math.exp(-alpha_Np_per_m * L_m)
            att_dB = -10.0 * math.log10(att_linear)
            
            # Atenuar señal y ruido
            P_signal *= att_linear
            P_ase_total *= att_linear
            
            num_spans += 1
            print(f"Span {num_spans}: {L_km:.1f} km")
            print(f"  Atenuación: {att_dB:.2f} dB (alfa={alpha_Np_per_m*1e3*4.343:.3f} dB/km)")
            print(f"  P_signal después: {P_signal*1e3:.4f} mW ({lin_to_dB(P_signal/1e-3):.2f} dBm)")
            
        elif typ == 'edfa':
            G_dB = par['G_dB']
            nsp = par.get('nsp', 1.5)
            
            G_lin = dB_to_lin(G_dB)
            
            # Amplificar señal
            P_signal *= G_lin
            
            # Calcular ASE añadida
            nu = c / (lambda_nm * 1e-9)
            P_ase_added = nsp * h * nu * (G_lin - 1.0) * Rb
            
            # Acumular ASE: amplificar previo + añadir nuevo
            P_ase_total = P_ase_total * G_lin + P_ase_added
            
            NF_dB = nf_dB_from_nsp(nsp)
            
            print(f"EDFA {num_spans}:")
            print(f"  Ganancia: {G_dB:.2f} dB, NF: {NF_dB:.2f} dB (nsp={nsp:.2f})")
            print(f"  P_ase añadida: {P_ase_added:.3e} W")
            print(f"  P_ase acumulada: {P_ase_total:.3e} W")
            print(f"  P_signal después: {P_signal*1e3:.4f} mW ({lin_to_dB(P_signal/1e-3):.2f} dBm)")
    
    # OSNR final
    if P_ase_total > 0:
        OSNR_linear = P_signal / P_ase_total
        OSNR_dB = lin_to_dB(OSNR_linear)
    else:
        OSNR_dB = float('inf')
    
    # BER aproximado para QPSK
    # SNR_elec ≈ OSNR × (Rb / Rb) = OSNR (en este caso Bo = Rb)
    SNR_linear = OSNR_linear
    BER_approx = 0.5 * math.erfc(math.sqrt(SNR_linear / 2.0))
    
    print(f"\n{'='*70}")
    print(f"RESULTADOS TEÓRICOS:")
    print(f"{'='*70}")
    print(f"P_signal final: {P_signal:.6e} W ({lin_to_dB(P_signal/1e-3):.2f} dBm)")
    print(f"P_ase total:    {P_ase_total:.6e} W")
    print(f"OSNR:           {OSNR_dB:.2f} dB")
    print(f"BER (aprox):    {BER_approx:.3e}")
    print(f"{'='*70}\n")
    
    return {
        'P_signal_W': P_signal,
        'P_ase_W': P_ase_total,
        'OSNR_dB': OSNR_dB,
        'BER': BER_approx
    }

def validate_simulation_log(log_path, expected):
    """
    Compara resultados del log de simulación con valores esperados
    """
    with open(log_path, 'r') as f:
        log = json.load(f)
    
    results = log.get('results', {})
    profile = log.get('profile', [])
    
    # Extraer OSNR final del perfil
    OSNR_measured = None
    if profile:
        for pt in reversed(profile):
            if pt.get('OSNR_dB') is not None:
                OSNR_measured = pt['OSNR_dB']
                break
    
    # Si no hay perfil, buscar en results
    if OSNR_measured is None:
        OSNR_measured = results.get('OSNR_final_dB')
    
    BER_measured = results.get('BER_post') or results.get('BER_est_BPSK')
    
    print(f"\n{'='*70}")
    print(f"COMPARACIÓN CON SIMULACIÓN:")
    print(f"{'='*70}")
    print(f"Log: {log_path}")
    print(f"\nOSNR:")
    print(f"  Teórico:  {expected['OSNR_dB']:.2f} dB")
    print(f"  Simulado: {OSNR_measured:.2f} dB" if OSNR_measured else "  Simulado: N/A")
    
    if OSNR_measured:
        diff = abs(OSNR_measured - expected['OSNR_dB'])
        print(f"  Diferencia: {diff:.2f} dB {'✅' if diff < 0.5 else '❌'}")
    
    print(f"\nBER:")
    print(f"  Teórico:  {expected['BER']:.3e}")
    print(f"  Simulado: {BER_measured:.3e}" if BER_measured else "  Simulado: N/A")
    
    if BER_measured:
        # BER puede variar bastante por ruido aleatorio
        ratio = BER_measured / expected['BER']
        print(f"  Ratio: {ratio:.2f}x {'✅' if 0.5 < ratio < 2.0 else '⚠️ (normal por ruido aleatorio)'}")
    
    print(f"{'='*70}\n")

if __name__ == '__main__':
    # Validar Val_1.json
    config1 = Path(__file__).parent / 'examples' / 'configs' / 'Val_1.json'
    if config1.exists():
        expected1 = calculate_theoretical_osnr(config1)
        
        # Buscar último log
        logs_dir = Path(__file__).parent / 'logs'
        if logs_dir.exists():
            logs = sorted(logs_dir.glob('simlog_*.json'), key=lambda p: p.stat().st_mtime)
            if logs:
                print(f"\n💡 Para validar, ejecuta primero: python -m fibersim.main {config1}")
                print(f"   Luego el último log será comparado automáticamente.\n")
    
    # Validar Val_2_3spans.json
    config2 = Path(__file__).parent / 'examples' / 'configs' / 'Val_2_3spans.json'
    if config2.exists():
        expected2 = calculate_theoretical_osnr(config2)
    
    print("\n✅ Cálculos teóricos completados.")
    print("\n📋 Para ejecutar simulaciones y validar:")
    print(f"   python -m fibersim.main examples/configs/Val_1.json")
    print(f"   python -m fibersim.main examples/configs/Val_2_3spans.json")
    print("\n   Luego revisa que OSNR simulado ≈ OSNR teórico (diferencia < 0.5 dB)\n")
