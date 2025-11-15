import json
from math import log10

h = 6.62607015e-34
nu = 3e8 / 1550e-9
Rs = 32e9

log = json.load(open('logs/simlog_2025-11-06_13-12-16.json'))

print("="*70)
print("SIMULACION DE ACUMULACION DE RUIDO ASE")
print("="*70)
print("\nEl ruido ASE se debe acumular considerando que:")
print("  1. Cada EDFA amplifica el ruido previo + añade ruido nuevo")
print("  2. Cada fibra atenúa el ruido")
print("\nFórmula correcta:")
print("  P_ase_nuevo = P_ase_previo × G + Pase_añadido")
print("\n" + "="*70)

P_ase = 0.0

for i, b in enumerate(log['chain'], 1):
    if b['type'] == 'edfa':
        G_dB = b['par']['G_dB']
        G_lin = 10**(G_dB/10)
        nsp = b['par']['nsp']
        
        # Ruido ASE añadido por este EDFA
        Pase_added = nsp * h * nu * (G_lin - 1) * Rs
        
        # El EDFA amplifica el ruido previo Y añade ruido nuevo
        P_ase_before = P_ase
        P_ase = P_ase * G_lin + Pase_added
        
        print(f"Paso {i} (EDFA): G={G_dB}dB, nsp={nsp}")
        print(f"  Ruido previo: {P_ase_before:.3e} W")
        print(f"  Ruido previo amplificado: {P_ase_before*G_lin:.3e} W")
        print(f"  Ruido añadido: {Pase_added:.3e} W")
        print(f"  TOTAL: {P_ase:.3e} W")
        
    elif b['type'] == 'fiber':
        alpha = b['par']['alpha']
        L = b['par']['L']
        # Atenuación en factor lineal
        att_dB = alpha * L * 4.343  # alpha está en Np/m, convertir a dB
        att_lin = 10**(-att_dB/10)
        
        P_ase_before = P_ase
        P_ase = P_ase * att_lin
        
        print(f"Paso {i} (Fibra): L={L/1000:.1f}km, alpha={alpha*1000:.3f}dB/km")
        print(f"  Atenuacion: {att_dB:.2f} dB")
        print(f"  Ruido antes: {P_ase_before:.3e} W")
        print(f"  Ruido después: {P_ase:.3e} W")

print("\n" + "="*70)
print("RESULTADO FINAL")
print("="*70)

P_signal = log['result']['Pmean_W']
OSNR_correct_dB = 10 * log10(P_signal / P_ase)
OSNR_reported_dB = log['result']['OSNR_final_dB']

print(f"\nP_ase acumulado (método correcto): {P_ase:.3e} W")
print(f"P_signal: {P_signal:.3e} W")
print(f"\nOSNR (método correcto) = 10×log10({P_signal:.3e} / {P_ase:.3e})")
print(f"OSNR correcto = {OSNR_correct_dB:.2f} dB")
print(f"OSNR reportado = {OSNR_reported_dB:.2f} dB")
print(f"\nDiferencia: {abs(OSNR_correct_dB - OSNR_reported_dB):.2f} dB")

if abs(OSNR_correct_dB - OSNR_reported_dB) < 0.5:
    print("\nOK: El simulador calcula el OSNR correctamente!")
else:
    print(f"\nERROR: Hay {abs(OSNR_correct_dB - OSNR_reported_dB):.1f} dB de discrepancia")
    print("El simulador probablemente NO está amplificando el ruido previo en los EDFAs")
