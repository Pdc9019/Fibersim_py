import json
from math import erfc, sqrt, log10

log = json.load(open('logs/simlog_2025-11-06_13-12-16.json'))

print("="*70)
print("VERIFICACIÓN DE LA CORRECCIÓN: OSNR → BER")
print("="*70)

OSNR_dB = log['result']['OSNR_final_dB']
BER_measured = log['result']['BER_post']
M = log['global']['M']
Rb = log['global']['Rb']

print(f"\n📊 DATOS DE LA SIMULACIÓN:")
print(f"  • OSNR medido: {OSNR_dB:.2f} dB")
print(f"  • BER medido: {BER_measured:.3e}")
print(f"  • Modulación: QPSK (M={M})")
print(f"  • Tasa símbolos Rb: {Rb/1e9:.0f} GHz")

print("\n" + "="*70)
print("MÉTODO ANTERIOR (INCORRECTO)")
print("="*70)
Bo_old = 12.5e9
OSNR_lin = 10**(OSNR_dB/10)
SNR_old_lin = OSNR_lin * (Bo_old / Rb)
SNR_old_dB = 10*log10(SNR_old_lin)
BER_old_teorico = 0.5 * erfc(sqrt(SNR_old_lin / 2.0))

print(f"  Asumía: OSNR medido en Bo = {Bo_old/1e9:.1f} GHz")
print(f"  Conversión: SNR = OSNR × (Bo/Rb) = {OSNR_dB:.2f} × {Bo_old/Rb:.3f}")
print(f"  SNR calculado: {SNR_old_dB:.2f} dB")
print(f"  BER teórico (QPSK): {BER_old_teorico:.3e}")
print(f"  BER medido: {BER_measured:.3e}")
print(f"  ❌ Discrepancia: {BER_measured/BER_old_teorico:.2e}x (absurda!)")

print("\n" + "="*70)
print("MÉTODO NUEVO (CORRECTO)")
print("="*70)
SNR_new_lin = OSNR_lin  # No hay conversión, OSNR ya está en BW = Rb
SNR_new_dB = OSNR_dB
BER_new_teorico = 0.5 * erfc(sqrt(SNR_new_lin / 2.0))

print(f"  Reconoce: OSNR medido en BW = Rb = {Rb/1e9:.0f} GHz")
print(f"  Por tanto: SNR ≈ OSNR (sin conversión)")
print(f"  SNR = {SNR_new_dB:.2f} dB")
print(f"  BER teórico (QPSK, AWGN): {BER_new_teorico:.3e}")
print(f"  BER medido: {BER_measured:.3e}")
print(f"  ✅ Ratio medido/teórico: {BER_measured/BER_new_teorico:.2f}x")

penalty_dB = -10*log10(BER_new_teorico / BER_measured)
print(f"  ✅ Penalización: {penalty_dB:.2f} dB")

print("\n" + "="*70)
print("INTERPRETACIÓN")
print("="*70)

if 0.5 <= BER_measured/BER_new_teorico <= 5:
    print("✅ La estrella ahora DEBE estar CERCA de la curva teórica")
    print(f"   Ratio {BER_measured/BER_new_teorico:.2f}x es razonable para un sistema real")
    print(f"   que incluye degradaciones ópticas no modeladas en AWGN puro")
elif BER_measured/BER_new_teorico < 0.5:
    print("🤔 BER medido es MEJOR que el teórico (inusual)")
    print("   Posibles causas: CDC muy efectivo, estimación optimista")
else:
    print("⚠️  BER medido es significativamente PEOR que el teórico")
    print(f"   Penalización de {penalty_dB:.1f} dB indica:")
    print("   • Dispersión cromática residual")
    print("   • Efectos no lineales (SPM, XPM)")
    print("   • PMD o jitter de fase")

print("\n" + "="*70)
print("COMPARACIÓN CON ESTÁNDARES")
print("="*70)

# OSNR requerido para BER < 10^-3 (QPSK)
for target_ber in [1e-3, 1e-4, 1e-5]:
    for osnr_test in range(5, 40):
        snr_lin_test = 10**(osnr_test/10)
        ber_test = 0.5 * erfc(sqrt(snr_lin_test / 2.0))
        if ber_test <= target_ber:
            print(f"  BER < {target_ber:.0e} requiere OSNR ≥ {osnr_test:.0f} dB")
            break

print(f"\n  Tu sistema:")
print(f"    OSNR = {OSNR_dB:.1f} dB")
print(f"    BER = {BER_measured:.2e}")

# Calcular margen
osnr_req_1e3 = None
for osnr_test in range(5, 40):
    snr_lin_test = 10**(osnr_test/10)
    ber_test = 0.5 * erfc(sqrt(snr_lin_test / 2.0))
    if ber_test <= 1e-3:
        osnr_req_1e3 = osnr_test
        break

if osnr_req_1e3:
    margin = OSNR_dB - osnr_req_1e3
    print(f"    Margen sobre umbral FEC: +{margin:.1f} dB {'✅' if margin > 0 else '❌'}")

print("\n" + "="*70)
print("CONCLUSIÓN")
print("="*70)
print("""
Con la corrección aplicada:
  ✅ El punto ⭐ ahora debe aparecer SOBRE o MUY CERCA de la curva teórica
  ✅ Es normal que esté ligeramente por ENCIMA (peor BER) debido a degradaciones
     ópticas reales que no están en el modelo AWGN puro
  ✅ Una separación de 1-3 dB es típica en sistemas reales
  
Recarga la GUI y verifica que la estrella esté alineada con la curva azul.
""")
