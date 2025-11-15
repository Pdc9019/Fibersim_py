import json
from math import erfc, sqrt, log10

# Cargar última simulación
log = json.load(open('logs/simlog_2025-11-06_13-12-16.json'))

print("="*70)
print("ANÁLISIS: ¿Por qué el BER medido está tan lejos de la curva teórica?")
print("="*70)

# Parámetros del sistema
OSNR_measured_dB = log['result']['OSNR_final_dB']
BER_measured = log['result']['BER_post']
M = log['global']['M']
Rb = log['global']['Rb']
mod = log['global']['mod']

print(f"\n📊 VALORES MEDIDOS EN LA SIMULACIÓN:")
print(f"  • OSNR final: {OSNR_measured_dB:.2f} dB")
print(f"  • BER medido: {BER_measured:.3e}")
print(f"  • Modulación: {mod} (M={M})")
print(f"  • Tasa de símbolos: {Rb/1e9:.0f} Gbaud")

# Parámetros para conversión OSNR → SNR
Bo_Hz = 12.5e9  # Ancho de banda óptico de referencia (0.1 nm @ 1550 nm)
print(f"\n🔧 PARÁMETROS DE CONVERSIÓN:")
print(f"  • Bo (ancho banda óptico): {Bo_Hz/1e9:.1f} GHz")
print(f"  • Rb (tasa símbolos): {Rb/1e9:.1f} Gbaud")
print(f"  • Relación Bo/Rb: {Bo_Hz/Rb:.3f}")

# Conversión OSNR → SNR
OSNR_lin = 10**(OSNR_measured_dB/10)
SNR_lin = OSNR_lin * (Bo_Hz / Rb)
SNR_dB = 10*log10(SNR_lin)

print(f"\n📐 CONVERSIÓN OSNR → SNR:")
print(f"  • OSNR lineal: {OSNR_lin:.2f}")
print(f"  • SNR = OSNR × (Bo/Rb) = {OSNR_lin:.2f} × {Bo_Hz/Rb:.3f}")
print(f"  • SNR lineal: {SNR_lin:.2f}")
print(f"  • SNR en dB: {SNR_dB:.2f} dB")

# BER teórico para este SNR (AWGN)
if M == 2:  # BPSK
    BER_teorico_AWGN = 0.5 * erfc(sqrt(SNR_lin))
    formula = "BER = 0.5 × erfc(√SNR)"
elif M == 4:  # QPSK
    BER_teorico_AWGN = 0.5 * erfc(sqrt(SNR_lin / 2.0))
    formula = "BER = 0.5 × erfc(√(SNR/2))"
elif M == 16:  # 16-QAM
    BER_teorico_AWGN = 0.75 * erfc(sqrt(SNR_lin / 10.0))
    formula = "BER = 0.75 × erfc(√(SNR/10))"
else:
    BER_teorico_AWGN = 0.5 * erfc(sqrt(SNR_lin / 2.0))
    formula = "BER aprox"

print(f"\n📚 BER TEÓRICO ({mod}, canal AWGN puro):")
print(f"  • Fórmula: {formula}")
print(f"  • BER teórico: {BER_teorico_AWGN:.3e}")

# Comparación
ratio = BER_measured / BER_teorico_AWGN
degradation_dB = 10 * log10(ratio)

print(f"\n⚠️  COMPARACIÓN:")
print(f"  • BER medido:   {BER_measured:.3e}")
print(f"  • BER teórico:  {BER_teorico_AWGN:.3e}")
print(f"  • Ratio (medido/teórico): {ratio:.2f}x")
print(f"  • Degradación: {degradation_dB:.2f} dB")

if ratio > 2:
    print(f"\n❌ PROBLEMA DETECTADO: BER medido es {ratio:.1f}x MAYOR que el teórico")
    print("   Esto NO debería ocurrir en un canal AWGN ideal.")
else:
    print(f"\n✅ BER medido está cerca del teórico (variación < 2x)")

# Posibles causas
print("\n"+"="*70)
print("🔍 POSIBLES CAUSAS DE LA DEGRADACIÓN")
print("="*70)

print("\n1️⃣ CONVERSIÓN OSNR → SNR:")
print("   ┌─ La fórmula SNR = OSNR × (Bo/Rb) asume:")
print("   │  • OSNR medido en ancho de banda Bo = 12.5 GHz (0.1 nm)")
print("   │  • Ruido blanco gaussiano distribuido uniformemente")
print("   │  • Receptor limitado por ancho de banda Rb")
print("   └─ ¿Estamos usando el Bo correcto?")

# Verificar si el OSNR está calculado correctamente
print("\n2️⃣ CÁLCULO DE OSNR EN LA SIMULACIÓN:")
print(f"   • OSNR se calcula como: 10×log10(P_signal / P_ase)")
print(f"   • P_ase es el ruido ASE acumulado de los {len([b for b in log['chain'] if b['type']=='edfa'])} EDFAs")
print(f"   • ¿El P_ase incluye el ancho de banda de referencia correcto?")

# Analizar la dispersión cromática
D_total_ps2 = sum(s['par']['L']*s['par']['beta2'] for s in log['chain'] if s['type']=='fiber') * 1e12
print("\n3️⃣ EFECTOS DE DISPERSIÓN CROMÁTICA:")
print(f"   • Dispersión total acumulada: {D_total_ps2:.1f} ps²")
if abs(D_total_ps2) < 10:
    print("   • ✅ Dispersión casi nula → No degrada BER")
    print("   • ⚠️ Esto es sospechoso - fibras reales tienen β₂ ≠ 0")
else:
    print(f"   • Dispersión significativa → CDC debe compensar")
    print(f"   • ¿CDC está funcionando correctamente?")

# Analizar efectos no lineales
gamma_avg = sum(s['par']['L']*s['par']['gamma'] for s in log['chain'] if s['type']=='fiber') / sum(s['par']['L'] for s in log['chain'] if s['type']=='fiber')
P_avg_W = log['result']['Pmean_W']
P_avg_dBm = 10*log10(P_avg_W*1000)
print("\n4️⃣ EFECTOS NO LINEALES:")
print(f"   • γ promedio: {gamma_avg*1000:.2f} /W/km")
print(f"   • Potencia promedio: {P_avg_W*1000:.3f} mW = {P_avg_dBm:.2f} dBm")
L_total_km = log['result']['Lcum_m']/1000
L_eff = 20  # Longitud efectiva típica
phi_NL = gamma_avg * P_avg_W * L_eff * 1000  # Fase no lineal
print(f"   • Fase no lineal estimada: {phi_NL:.4f} rad")
if phi_NL > 0.1:
    print(f"   • ⚠️ Efectos no lineales pueden degradar BER")
else:
    print(f"   • ✅ Efectos no lineales despreciables")

# Verificar el ancho de banda de referencia del OSNR
print("\n5️⃣ ANCHO DE BANDA DE REFERENCIA (CRÍTICO):")
print(f"   Nuestra fórmula usa Bo = {Bo_Hz/1e9:.1f} GHz")
print(f"   Estándares comunes:")
print(f"   • ITU-T G.698.2: Bo = 12.5 GHz (0.1 nm @ 1550 nm) ✓")
print(f"   • Algunos equipos: Bo = 0.5 nm = 62.5 GHz")
print(f"   • DWDM spacing: 50 GHz típico")
print("")
print(f"   Si el simulador usa Bo diferente en el cálculo de P_ase:")
print(f"   → OSNR estaría mal escalado")
print(f"   → Conversión OSNR→SNR daría valores incorrectos")

# Prueba: recalcular con diferentes Bo
print("\n6️⃣ EXPERIMENTO: BER con diferentes anchos de banda:")
for Bo_test_GHz in [12.5, 25, 50, 62.5]:
    Bo_test_Hz = Bo_test_GHz * 1e9
    SNR_test_lin = OSNR_lin * (Bo_test_Hz / Rb)
    if M == 4:  # QPSK
        BER_test = 0.5 * erfc(sqrt(SNR_test_lin / 2.0))
    else:
        BER_test = 0.5 * erfc(sqrt(SNR_test_lin))
    
    diff_percent = abs(BER_test - BER_measured) / BER_measured * 100
    marker = "⭐" if diff_percent < 10 else "  "
    
    print(f"   {marker} Bo = {Bo_test_GHz:5.1f} GHz → SNR = {10*log10(SNR_test_lin):6.2f} dB → BER = {BER_test:.3e} (dif: {diff_percent:5.1f}%)")

print("\n" + "="*70)
print("🎯 DIAGNÓSTICO Y RECOMENDACIONES")
print("="*70)

# Revisar código de cálculo de OSNR
print("\n📝 ACCIONES A TOMAR:")
print("  1. Revisar cálculo de P_ase en edfa.py")
print("     → Verificar que use el ancho de banda correcto")
print("  2. Revisar cálculo de OSNR en chain.py")
print("     → OSNR = P_signal / P_ase_in_Bo")
print("  3. Verificar que Bo en GUI coincida con Bo en simulación")
print("  4. Considerar que BER 'real' incluye:")
print("     • Dispersión residual (si CDC no es perfecto)")
print("     • Efectos no lineales (SPM, XPM)")
print("     • Jitter de fase del láser")
print("     • PMD (Polarization Mode Dispersion)")
print("     • Ruido del receptor")
print("")
print("  💡 Es NORMAL que BER medido > BER teórico AWGN")
print("     pero la diferencia debería ser ~1-3 dB, no más.")
