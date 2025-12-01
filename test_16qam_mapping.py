"""Test rápido del mapeo/demapeo de 16-QAM"""
import numpy as np
import sys
sys.path.insert(0, 'src')

from fibersim.core.modem import map_bits_to_symbols, _symbols_to_bits, slice_symbols

# Test: generar todos los 16 símbolos posibles
print("=" * 60)
print("TEST DE MAPEO/DEMAPEO 16-QAM")
print("=" * 60)

# Generar todos los posibles 4-bit patterns (16 combinaciones)
all_bits = []
for i in range(16):
    # i en binario de 4 bits
    bits = [(i >> (3-j)) & 1 for j in range(4)]
    all_bits.extend(bits)

all_bits_np = np.array(all_bits, dtype=np.uint8)
print(f"\nBits de entrada (16 símbolos × 4 bits): {len(all_bits_np)} bits")
print(f"Primeros 16 bits (símbolo 0): {all_bits_np[:4]}")
print(f"Bits 4-7 (símbolo 1): {all_bits_np[4:8]}")

# Mapear a símbolos
syms = map_bits_to_symbols(all_bits_np, M=16, xp=np)
print(f"\nSímbolos generados: {len(syms)} símbolos")

# Mostrar mapeo
print("\n" + "=" * 60)
print("MAPEO TX (bits → símbolos):")
print("=" * 60)
for i in range(16):
    bits_i = all_bits_np[i*4:(i+1)*4]
    sym_i = syms[i]
    print(f"Símbolo {i:2d}: bits={bits_i} → sym={sym_i.real:+.3f}{sym_i.imag:+.3f}j  "
          f"(escala: {sym_i.real*np.sqrt(10):+.1f}{sym_i.imag*np.sqrt(10):+.1f}j)")

# Ahora demap de vuelta
print("\n" + "=" * 60)
print("SLICE + DEMAPEO RX (símbolos → bits):")
print("=" * 60)

# Slice (debería ser identidad para símbolos perfectos)
syms_sliced = slice_symbols(syms, "16QAM")
bits_recovered = _symbols_to_bits(syms_sliced, "16QAM")

print(f"Bits recuperados: {len(bits_recovered)} bits")

# Comparar
errors = 0
for i in range(16):
    bits_tx = all_bits_np[i*4:(i+1)*4]
    bits_rx = bits_recovered[i*4:(i+1)*4]
    match = "✓" if np.array_equal(bits_tx, bits_rx) else "✗"
    if not np.array_equal(bits_tx, bits_rx):
        errors += 1
    sym_i = syms[i]
    print(f"Símbolo {i:2d}: TX_bits={bits_tx} → RX_bits={bits_rx} {match}  "
          f"(sym={sym_i.real*np.sqrt(10):+.1f}{sym_i.imag*np.sqrt(10):+.1f}j)")

print("\n" + "=" * 60)
print(f"RESULTADO: {16-errors}/16 símbolos correctos ({errors} errores)")
print("=" * 60)

if errors == 0:
    print("✓ MAPEO/DEMAPEO FUNCIONA CORRECTAMENTE")
else:
    print("✗ HAY ERRORES EN EL MAPEO/DEMAPEO")
    sys.exit(1)
