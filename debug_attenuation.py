import json
from math import log10

log = json.load(open('logs/simlog_2025-11-06_15-10-27.json'))

# Constantes
h = 6.62607015e-34
nu = 3e8 / 1550e-9
Rs = log['global']['Rb']

print("="*70)
print("DEBUG: Verificar cálculo de atenuación en chain.py")
print("="*70)

# Simular exactamente lo que hace chain.py
P_ase_total = 0.0
step_const_m = 5000.0  # 5 km

for i, b in enumerate(log['chain'], 1):
    if b['type'] == 'fiber':
        L = b['par']['L']
        alpha = b['par']['alpha']
        
        print(f"\n{i}. Fibra: L={L/1000:.1f} km, alpha={alpha*1000:.3f} dB/km")
        
        # chain.py hace un loop interno
        Lpend = L
        step_num = 0
        while Lpend > 0:
            step = min(step_const_m, Lpend)
            step_num += 1
            
            # Atenuación en este step
            att_dB = alpha * step * 4.343
            att_lin = 10**(-att_dB/10)
            
            P_ase_before = P_ase_total
            P_ase_total = P_ase_total * att_lin
            
            if step_num == 1 or Lpend == step:  # Primer y último step
                print(f"   Step {step_num}: L={step/1000:.1f}km, att={att_dB:.3f}dB")
                print(f"     P_ase: {P_ase_before:.3e} -> {P_ase_total:.3e} W")
            
            Lpend -= step
        
        print(f"   Total: P_ase = {P_ase_total:.3e} W")
        
    elif b['type'] == 'edfa':
        G_dB = b['par']['G_dB']
        G_lin = 10**(G_dB/10)
        nsp = b['par']['nsp']
        
        Pase_added = nsp * h * nu * (G_lin - 1) * Rs
        
        P_ase_before = P_ase_total
        P_ase_total = P_ase_total * G_lin + Pase_added
        
        print(f"\n{i}. EDFA: G={G_dB}dB, nsp={nsp}")
        print(f"   P_ase antes: {P_ase_before:.3e} W")
        print(f"   Amplificado: {P_ase_before*G_lin:.3e} W")
        print(f"   Añadido: {Pase_added:.3e} W")
        print(f"   Total: {P_ase_total:.3e} W")

P_signal = log['result']['Pmean_W']
OSNR_manual = 10 * log10(P_signal / P_ase_total) if P_ase_total > 0 else float('inf')

print("\n" + "="*70)
print("RESULTADO FINAL")
print("="*70)
print(f"P_signal: {P_signal:.3e} W")
print(f"P_ase total: {P_ase_total:.3e} W")
print(f"OSNR manual: {OSNR_manual:.2f} dB")
print(f"OSNR reportado: {log['result']['OSNR_final_dB']:.2f} dB")
print(f"Diferencia: {abs(OSNR_manual - log['result']['OSNR_final_dB']):.2f} dB")
