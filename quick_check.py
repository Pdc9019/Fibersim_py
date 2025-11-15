import json
from math import log10

# Usar el log más reciente
log = json.load(open('logs/simlog_2025-11-06_15-20-06.json'))

# Constantes
h = 6.62607015e-34
nu = 3e8 / 1550e-9
Rs = log['global']['Rb']

# Calcular P_ase manualmente con AMPLIFICACIÓN correcta
P_ase_total = 0.0

for b in log['chain']:
    if b['type'] == 'edfa':
        G_dB = b['par']['G_dB']
        G_lin = 10**(G_dB/10)
        nsp = b['par']['nsp']
        Pase_added = nsp * h * nu * (G_lin - 1) * Rs
        # MÉTODO CORRECTO: amplificar ruido previo
        P_ase_total = P_ase_total * G_lin + Pase_added
    elif b['type'] == 'fiber':
        alpha = b['par']['alpha']
        L = b['par']['L']
        att_dB = alpha * L * 4.343
        att_lin = 10**(-att_dB/10)
        P_ase_total = P_ase_total * att_lin

P_signal = log['result']['Pmean_W']
OSNR_manual = 10 * log10(P_signal / P_ase_total)
OSNR_reported = log['result']['OSNR_final_dB']

print("="*70)
print("VERIFICACIÓN: ¿Se aplicó la corrección?")
print("="*70)
print(f"\nSimulación: {log['date']}")
print(f"\nP_signal: {P_signal:.3e} W")
print(f"P_ase (método correcto): {P_ase_total:.3e} W")
print(f"\nOSNR calculado manualmente: {OSNR_manual:.2f} dB")
print(f"OSNR reportado por simulador: {OSNR_reported:.2f} dB")
print(f"Diferencia: {abs(OSNR_manual - OSNR_reported):.2f} dB")

if abs(OSNR_manual - OSNR_reported) < 0.5:
    print("\n✅ ÉXITO: La corrección SE APLICÓ correctamente!")
    print("   El simulador ahora amplifica el ruido ASE en los EDFAs")
else:
    print(f"\n❌ ERROR: La corrección NO se aplicó")
    print(f"   Diferencia de {abs(OSNR_manual - OSNR_reported):.1f} dB indica que el código viejo sigue activo")
    print("\n   Posibles causas:")
    print("   1. Python tiene módulos en cache (necesita importlib.reload)")
    print("   2. Streamlit no reinició completamente")
    print("   3. Los cambios en chain.py no se guardaron")
