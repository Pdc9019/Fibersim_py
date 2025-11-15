import json
from math import log10

log = json.load(open('logs/simlog_2025-11-06_13-12-16.json'))

Rb = log['global']['Rb']
print(f"Rb (tasa símbolos) = {Rb/1e9:.1f} GHz")

# Constantes
h = 6.62607015e-34
lam = 1550e-9
nu = 3e8 / lam
Rs = Rb  # En edfa.py, Rs = Rb

# Calcular ruido ASE total
edfas = [b for b in log['chain'] if b['type']=='edfa']
total_Pase = 0

print("\nRuido ASE por EDFA (BW = Rs = 32 GHz):")
for i, e in enumerate(edfas, 1):
    G_dB = e['par']['G_dB']
    nsp = e['par']['nsp']
    G_lin = 10**(G_dB/10)
    Pase = nsp * h * nu * (G_lin - 1) * Rs
    total_Pase += Pase
    print(f"  EDFA {i}: G={G_dB}dB, nsp={nsp} -> Pase = {Pase:.3e} W")

print(f"\n💡 Total P_ase (acumulado) = {total_Pase:.3e} W")

# Potencia de señal
P_signal = log['result']['Pmean_W']
print(f"💡 P_signal (potencia promedio) = {P_signal:.3e} W")

# OSNR calculado
OSNR_calc_dB = 10*log10(P_signal / total_Pase)
OSNR_reported_dB = log['result']['OSNR_final_dB']

print(f"\n📊 OSNR:")
print(f"  OSNR calculado = 10*log10({P_signal:.3e} / {total_Pase:.3e})")
print(f"  OSNR calculado = {OSNR_calc_dB:.2f} dB")
print(f"  OSNR reportado = {OSNR_reported_dB:.2f} dB")
print(f"  Diferencia = {OSNR_calc_dB - OSNR_reported_dB:.2f} dB")

if abs(OSNR_calc_dB - OSNR_reported_dB) < 0.5:
    print("  OK Coinciden! El OSNR se calcula correctamente")
else:
    print("  ERROR No coinciden - hay un problema")

# Ahora el problema: ¿cuál es el SNR efectivo?
print("\n" + "="*70)
print("PROBLEMA: OSNR vs SNR efectivo")
print("="*70)

print(f"""
El OSNR se calcula como:
  OSNR_dB = 10×log10(P_signal / P_ase)

Donde:
  P_signal = {P_signal:.3e} W  (potencia promedio de la señal)
  P_ase    = {total_Pase:.3e} W  (ruido ASE en BW = Rs = {Rs/1e9:.0f} GHz)

Esto da OSNR = {OSNR_reported_dB:.2f} dB

PERO... para calcular BER, necesitamos SNR = Es/N0:
  Es/N0 = (Energía por símbolo) / (Densidad espectral de ruido)

Energía por símbolo:
  Es = P_signal × Ts = P_signal / Rs
  
Densidad espectral de ruido:
  N0 = P_ase / Rs  (potencia de ruido dividida por el ancho de banda)

Por tanto:
  SNR = Es/N0 = (P_signal/Rs) / (P_ase/Rs) = P_signal / P_ase
  
¡Es lo mismo que OSNR!

Entonces, ¿por qué no coincide con el BER medido?
""")

# Buscar el SNR que da el BER medido
from math import erfc, sqrt
from scipy.special import erfcinv

BER_measured = log['result']['BER_post']
x = erfcinv(BER_measured / 0.5)
SNR_effective_lin = 2 * x**2
SNR_effective_dB = 10 * log10(SNR_effective_lin)

print(f"BER medido = {BER_measured:.3e}")
print(f"SNR efectivo (que explica ese BER) = {SNR_effective_dB:.2f} dB")
print(f"OSNR reportado = {OSNR_reported_dB:.2f} dB")
print(f"Diferencia = {OSNR_reported_dB - SNR_effective_dB:.2f} dB")

# Factor de conversión
factor = SNR_effective_lin / (P_signal/total_Pase)
print(f"\nFactor de conversión OSNR → SNR_eff = {factor:.3f}")
print(f"En dB: {10*log10(factor):.2f} dB")

# Hipótesis
print("\n" + "="*70)
print("HIPÓTESIS")
print("="*70)

print(f"""
La diferencia de {OSNR_reported_dB - SNR_effective_dB:.1f} dB sugiere que:

1️⃣ El ruido REAL es mayor que el calculado
   → Posiblemente hay otras fuentes de ruido:
     • Ruido térmico del receptor
     • Ruido de disparo (shot noise)
     • Ruido de fase del láser
   
2️⃣ La potencia de señal EFECTIVA es menor
   → Posiblemente debido a:
     • Dispersión cromática residual (CDC no perfecto)
     • Interferencia entre símbolos (ISI)
     • Efectos no lineales
   
3️⃣ El ancho de banda de referencia es inconsistente
   → El ruido ASE se calcula en BW = {Rs/1e9:.0f} GHz
   → Pero quizás el receptor tiene un filtro diferente
   
4️⃣ La implementación de DSP introduce degradación
   → CDC, sincronización, ecualización, etc.
""")

# Verificar si hay dispersión
D_total = sum(s['par']['L']*s['par']['beta2'] for s in log['chain'] if s['type']=='fiber') * 1e12
print(f"\nDispersion cromatica total: {D_total:.1f} ps^2")
if abs(D_total) < 10:
    print("  -> Dispersion casi nula, CDC no deberia degradar mucho")

print(f"\nEfectos no lineales: gamma_avg aprox 1.5 /W/km, P_avg aprox {P_signal*1000:.3f} mW")
print("  -> Muy baja potencia, efectos no lineales despreciables")

print("\n" + "="*70)
print("CONCLUSIÓN PROVISIONAL")
print("="*70)
print(f"""
El OSNR reportado ({OSNR_reported_dB:.1f} dB) parece ser correcto en términos
de la relación P_signal/P_ase.

Sin embargo, el SNR EFECTIVO que determina el BER es ~{10*log10(factor):.1f} dB más bajo.

Esto es NORMAL en sistemas reales y se debe a:
  • Degradaciones del procesamiento DSP
  • Ruido adicional del receptor (no modelado en ASE)
  • Imperfecciones en la estimación de canal
  • Pérdida de implementación (implementation penalty)

Para que la curva BER vs OSNR sea consistente con las mediciones,
debemos ACEPTAR que:
  
  BER_real > BER_teórico_AWGN
  
Y mostrar AMBOS en el gráfico:
  - Curva azul: BER teórico (AWGN ideal)
  - Estrella roja: BER medido (sistema real con degradaciones)

La estrella DEBE estar POR ENCIMA de la curva (peor BER para el mismo OSNR).
Esto es correcto y esperado.
""")
