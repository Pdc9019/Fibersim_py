from math import erfc, sqrt, log10

print("="*70)
print("ANÁLISIS: Fórmulas BER para QPSK")
print("="*70)

# SNR del sistema
SNR_dB = 24.2
SNR_lin = 10**(SNR_dB/10)

print(f"\nSNR = {SNR_dB} dB = {SNR_lin:.2f} (lineal)")

print("\n" + "="*70)
print("DIFERENTES DEFINICIONES DE SNR EN QPSK")
print("="*70)

# 1. SNR por símbolo (Es/N0)
print("\n1️⃣ SNR por SÍMBOLO (Es/N0):")
print("   BER = 0.5 × erfc(√(Es/N0 / 2))")
BER_1 = 0.5 * erfc(sqrt(SNR_lin / 2))
print(f"   Con Es/N0 = {SNR_dB} dB:")
print(f"   BER = {BER_1:.3e}")

# 2. SNR por BIT (Eb/N0) - QPSK tiene 2 bits/símbolo
print("\n2️⃣ SNR por BIT (Eb/N0):")
print("   Para QPSK: Es = 2×Eb (2 bits por símbolo)")
print("   Por tanto: Eb/N0 = Es/N0 / 2")
Eb_N0_lin = SNR_lin / 2
Eb_N0_dB = 10*log10(Eb_N0_lin)
print(f"   Eb/N0 = {SNR_dB} - 3 = {Eb_N0_dB:.2f} dB")
print("   BER = 0.5 × erfc(√(Eb/N0))")
BER_2 = 0.5 * erfc(sqrt(Eb_N0_lin))
print(f"   BER = {BER_2:.3e}")

# 3. SNR promedio de la señal
print("\n3️⃣ SNR PROMEDIO DE LA SEÑAL:")
print("   Si SNR_dB es la relación Potencia_señal / Potencia_ruido")
print("   en el ancho de banda completo (Rb = 32 GHz):")
print("   Entonces es equivalente a Es/N0")
print(f"   BER = 0.5 × erfc(√(SNR/2)) = {BER_1:.3e}")

print("\n" + "="*70)
print("¿CUÁL ES LA CORRECTA?")
print("="*70)

print("""
Depende de cómo se calcula el OSNR en edfa.py:

En edfa.py línea 19:
   Pase = nsp * h * nu * (G - 1) * Rs

Donde Rs = Rb = tasa de símbolos

Este es el ruido ASE en el ancho de banda de un solo canal (Rs).

Por tanto:
   OSNR_dB = 10×log10(P_signal / P_ase)
   
Donde:
   - P_signal = Potencia de la señal modulada
   - P_ase = Potencia del ruido ASE en BW = Rs

¿Esto corresponde a Es/No o a Eb/N0?

🔍 ANÁLISIS:
- P_signal es la potencia TOTAL de la señal (incluye ambas polarizaciones si es dual-pol)
- Rs es la tasa de SÍMBOLOS (no bits)
- Por tanto, OSNR_dB ≈ 10×log10(Es/N0)
- Para QPSK: BER = Q(√(2×Es/N0)) ≈ 0.5 × erfc(√(Es/N0 / 2))

Pero...
""")

# Verificar qué fórmula da el BER medido
BER_measured = 3.662e-04

print(f"\nBER medido en simulación: {BER_measured:.3e}")
print(f"\nProbemos diferentes interpretaciones:\n")

# Buscar qué SNR da el BER medido
for description, snr_formula in [
    ("Es/N0 directo", SNR_lin),
    ("Es/N0 / 2", SNR_lin / 2),
    ("Es/N0 * 2", SNR_lin * 2),
    ("Eb/N0 (= Es/N0 / 2)", SNR_lin / 2)
]:
    ber_test = 0.5 * erfc(sqrt(snr_formula / 2))
    diff = abs(ber_test - BER_measured) / BER_measured * 100
    marker = "⭐" if diff < 50 else "  "
    print(f"{marker} {description:20s}: SNR_eff = {10*log10(snr_formula):6.2f} dB → BER = {ber_test:.3e} (dif: {diff:6.1f}%)")

# Resolver inversamente: ¿qué SNR da BER = 3.662e-04?
# BER = 0.5 × erfc(√(SNR/2))
# erfc(x) = BER/0.5
# x = erfc_inv(BER/0.5)
# √(SNR/2) = x
# SNR = 2×x²

from scipy.special import erfcinv
x = erfcinv(BER_measured / 0.5)
SNR_required_lin = 2 * x**2
SNR_required_dB = 10 * log10(SNR_required_lin)

print(f"\n🎯 SNR requerido para BER = {BER_measured:.3e}:")
print(f"   SNR_req = {SNR_required_dB:.2f} dB")
print(f"   SNR_medido = {SNR_dB:.2f} dB")
print(f"   Diferencia: {SNR_dB - SNR_required_dB:.2f} dB")

if abs(SNR_dB - SNR_required_dB) < 3:
    print(f"\n✅ CONCLUSIÓN: La fórmula BER = 0.5×erfc(√(SNR/2)) es CORRECTA")
    print(f"   con SNR = OSNR (sin conversión adicional)")
else:
    factor_needed = SNR_required_lin / SNR_lin
    print(f"\n⚠️  Hay {SNR_dB - SNR_required_dB:.1f} dB de diferencia")
    print(f"   Factor de corrección necesario: {factor_needed:.3f}")
    print(f"   Esto sugiere que SNR_efectivo = OSNR × {factor_needed:.3f}")
