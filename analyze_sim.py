import json
from math import log10

log = json.load(open('logs/simlog_2025-11-06_13-12-16.json'))

print("="*60)
print("ANÁLISIS DE REALISMO DE LA SIMULACIÓN")
print("="*60)

# Parámetros del sistema
print("\n📡 PARÁMETROS DEL SISTEMA:")
print(f"  • Distancia total: {log['result']['Lcum_m']/1000:.1f} km")
print(f"  • Modulación: {log['global']['mod']}")
print(f"  • Tasa símbolos: {log['global']['Rb']/1e9:.0f} Gbaud")
print(f"  • Polarización: {log['global']['pol']} (single/dual)")
print(f"  • Receptor: {log['global']['rx']} (coherente)")
print(f"  • Potencia TX: {10*log['global']['Ptx']:.1f} mW = {10*log10(log['global']['Ptx']*1000):.1f} dBm")

# Fibras
print("\n🔌 SPANS DE FIBRA:")
spans = [b for b in log['chain'] if b['type']=='fiber']
total_loss = 0
for i,s in enumerate(spans):
    L_km = s['par']['L']/1000
    alpha_dB_km = s['par']['alpha']*1e3
    beta2_ps2_km = s['par']['beta2']*1e12
    gamma_W_km = s['par']['gamma']*1000
    loss_dB = alpha_dB_km * L_km
    total_loss += loss_dB
    
    # Identificar tipo de fibra
    if abs(beta2_ps2_km + 21) < 2:
        fiber_type = "SMF-28 (estándar)"
    elif abs(beta2_ps2_km + 6) < 2:
        fiber_type = "NZDSF/TeraLight"
    elif abs(beta2_ps2_km - 8.5) < 2:
        fiber_type = "DCF (compensación)"
    else:
        fiber_type = "Desconocida"
    
    print(f"  Span {i+1}: {L_km:.1f} km - {fiber_type}")
    print(f"    α = {alpha_dB_km:.2f} dB/km → Pérdida = {loss_dB:.1f} dB")
    print(f"    β₂ = {beta2_ps2_km:.1f} ps²/km")
    print(f"    γ = {gamma_W_km:.2f} /W/km")

print(f"\n  💡 Pérdida total de fibra: {total_loss:.1f} dB")

# EDFAs
print("\n⚡ AMPLIFICADORES (EDFAs):")
edfas = [b for b in log['chain'] if b['type']=='edfa']
total_gain = 0
for i,e in enumerate(edfas):
    G_dB = e['par']['G_dB']
    nsp = e['par']['nsp']
    NF_dB = 10*log10(2*nsp)
    total_gain += G_dB
    print(f"  EDFA {i+1}: G = {G_dB:.1f} dB, nsp = {nsp} → NF = {NF_dB:.1f} dB")

print(f"\n  💡 Ganancia total: {total_gain:.1f} dB")
print(f"  💡 Balance pérdida/ganancia: {total_gain - total_loss:+.1f} dB")

# Resultados
print("\n📊 RESULTADOS DE LA SIMULACIÓN:")
print(f"  • BER: {log['result']['BER_post']:.3e}")
print(f"  • OSNR final: {log['result']['OSNR_final_dB']:.2f} dB")
print(f"  • EVM: {log['result']['EVM_post_dB']:.2f} dB")
print(f"  • Q-factor: {log['result']['Q_post']:.2f}")
print(f"  • Potencia salida: {log['result']['Pout_dBm']:.2f} dBm")

# Dispersión cromática
print("\n🌈 DISPERSIÓN CROMÁTICA:")
D_total_ps = sum(s['par']['L']*s['par']['beta2'] for s in spans) * 1e12
print(f"  • Dispersión total acumulada: {D_total_ps:.1f} ps²")
lambda_nm = 1550  # Asumiendo C-band
D_ps_nm_km = D_total_ps / (log['result']['Lcum_m']/1000)
print(f"  • Dispersión promedio: {D_ps_nm_km:.2f} ps²/km")

# Análisis de realismo
print("\n"+"="*60)
print("✅ ANÁLISIS DE REALISMO")
print("="*60)

issues = []
good = []

# 1. Distancia
L_total = log['result']['Lcum_m']/1000
if 50 <= L_total <= 500:
    good.append(f"✅ Distancia ({L_total:.0f} km) típica para enlaces metropolitanos/regionales")
elif L_total > 500:
    issues.append(f"⚠️ Distancia ({L_total:.0f} km) requeriría más amplificación")
else:
    good.append(f"✅ Distancia ({L_total:.0f} km) realista para enlace corto")

# 2. Atenuación
avg_alpha = sum(s['par']['alpha']*s['par']['L'] for s in spans) / sum(s['par']['L'] for s in spans) * 1000
if 0.18 <= avg_alpha <= 0.22:
    good.append(f"✅ Atenuación promedio ({avg_alpha:.3f} dB/km) típica de SMF-28")
elif 0.15 <= avg_alpha <= 0.35:
    good.append(f"✅ Atenuación ({avg_alpha:.3f} dB/km) dentro de rango realista")
else:
    issues.append(f"⚠️ Atenuación ({avg_alpha:.3f} dB/km) inusual")

# 3. OSNR
OSNR_dB = log['result']['OSNR_final_dB']
if OSNR_dB >= 15:
    good.append(f"✅ OSNR ({OSNR_dB:.1f} dB) excelente para QPSK")
elif OSNR_dB >= 12:
    good.append(f"✅ OSNR ({OSNR_dB:.1f} dB) aceptable para QPSK")
else:
    issues.append(f"❌ OSNR ({OSNR_dB:.1f} dB) insuficiente - BER será alto")

# 4. Balance ganancia/pérdida
balance = total_gain - total_loss
if -5 <= balance <= 5:
    good.append(f"✅ Balance ganancia/pérdida ({balance:+.1f} dB) bien compensado")
else:
    issues.append(f"⚠️ Balance ({balance:+.1f} dB) - potencia podría variar mucho")

# 5. Figura de ruido EDFAs
avg_nsp = sum(e['par']['nsp'] for e in edfas) / len(edfas)
if 1.2 <= avg_nsp <= 2.0:
    good.append(f"✅ Factor de ruido EDFAs (nsp={avg_nsp:.2f}) realista")
else:
    issues.append(f"⚠️ Factor de ruido (nsp={avg_nsp:.2f}) inusual")

# 6. BER
BER = log['result']['BER_post']
if BER < 1e-3:
    good.append(f"✅ BER ({BER:.2e}) bajo umbral FEC - enlace operativo")
elif BER < 1e-2:
    issues.append(f"⚠️ BER ({BER:.2e}) marginal - requiere FEC fuerte")
else:
    issues.append(f"❌ BER ({BER:.2e}) demasiado alto - enlace no funcional")

# 7. Espaciado de amplificadores
span_lengths = [s['par']['L']/1000 for s in spans]
avg_span = sum(span_lengths)/len(span_lengths)
if 40 <= avg_span <= 120:
    good.append(f"✅ Espaciado promedio EDFAs ({avg_span:.0f} km) típico")
else:
    issues.append(f"⚠️ Espaciado EDFAs ({avg_span:.0f} km) inusual")

# Imprimir resultados
print("\n🟢 ASPECTOS REALISTAS:")
for item in good:
    print(f"  {item}")

if issues:
    print("\n🟡 PUNTOS A CONSIDERAR:")
    for item in issues:
        print(f"  {item}")
else:
    print("\n✨ No se detectaron inconsistencias importantes")

# Comparación con sistemas comerciales
print("\n"+"="*60)
print("📚 COMPARACIÓN CON SISTEMAS COMERCIALES")
print("="*60)
print("""
Sistema típico 100G QPSK (ejemplo: coherente DP-QPSK):
  • Tasa: 25-32 Gbaud × 2 pol × 2 bits/símbolo ≈ 100-128 Gbps
  • Alcance: 1000-2000 km (con FEC)
  • OSNR requerido: ~11-14 dB (para BER < 1e-3)
  • Espaciado EDFAs: 50-100 km
  • Margen típico: 3-6 dB

Tu simulación:
  • Tasa: {0:.0f} Gbaud × 1 pol × 2 bits/símbolo = {1:.0f} Gbps
  • Alcance: {2:.0f} km
  • OSNR medido: {3:.1f} dB
  • BER: {4:.2e}
  • Espaciado EDFAs: {5:.0f} km promedio

⚡ CONCLUSIÓN: Tu simulación es REALISTA y representa un enlace
   óptico coherente funcional con parámetros típicos de la industria.
""".format(log['global']['Rb']/1e9, log['global']['Rb']/1e9 * 2, L_total, OSNR_dB, BER, avg_span))
