"""
Test para verificar la potencia después del filtro RRC con upsampling.
"""
import numpy as np
from scipy import signal

def rrc_taps(beta, span, sps):
    """Generar taps RRC normalizados en energía."""
    N = 2 * span * sps + 1
    t = (np.arange(N) - N // 2) / sps
    h = np.zeros(N)
    for i, ti in enumerate(t):
        if ti == 0:
            h[i] = (1 + beta * (4 / np.pi - 1))
        elif abs(ti) == 1 / (4 * beta) and beta != 0:
            h[i] = (beta / np.sqrt(2)) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta)) +
                                           (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
        else:
            num = np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))
            den = np.pi * ti * (1 - (4 * beta * ti) ** 2)
            h[i] = num / den if den != 0 else 0
    # Normalizar a energía 1
    h = h / np.sqrt(np.sum(h ** 2))
    return h

# Parámetros
sps = 8
roll = 0.1
span = 10
Nsym = 10000

# Generar símbolos QPSK normalizados
np.random.seed(42)
bits = np.random.randint(0, 2, Nsym * 2)
I = 2 * bits[::2] - 1
Q = 2 * bits[1::2] - 1
syms = (I + 1j * Q) / np.sqrt(2)

print(f"Potencia símbolos: {np.mean(np.abs(syms)**2):.4f}")

# Upsample
up = np.zeros(syms.size * sps, dtype=complex)
up[::sps] = syms

print(f"Potencia upsampled: {np.mean(np.abs(up)**2):.4f}")
print(f"  (debería ser ~{1/sps:.4f} = 1/sps)")

# Filtrar con RRC
h = rrc_taps(beta=roll, span=span, sps=sps)
print(f"Energía filtro RRC: {np.sum(h**2):.4f}")

y = signal.lfilter(h, [1], up)

print(f"Potencia después RRC: {np.mean(np.abs(y)**2):.4f}")
print(f"  Factor respecto a símbolos: {np.mean(np.abs(y)**2) / np.mean(np.abs(syms)**2):.4f}")
print(f"  Factor respecto a upsampled: {np.mean(np.abs(y)**2) / np.mean(np.abs(up)**2):.4f}")

# Ahora con matched filter en RX (otro RRC)
y_rx = signal.lfilter(h, [1], y)
print(f"\nPotencia después matched filter: {np.mean(np.abs(y_rx)**2):.4f}")
print(f"  Factor respecto a símbolos: {np.mean(np.abs(y_rx)**2) / np.mean(np.abs(syms)**2):.4f}")
