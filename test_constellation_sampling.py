"""
Script de prueba para comparar constelaciones con diferentes niveles de muestreo.
Esto demuestra la diferencia entre mostrar todos los puntos sobremuestreados
vs mostrar solo 1 punto por símbolo.
"""

import numpy as np
import matplotlib.pyplot as plt

# Simular símbolos QPSK
np.random.seed(42)
n_symbols = 1000
qpsk_symbols = np.random.choice([1+1j, 1-1j, -1+1j, -1-1j], size=n_symbols) / np.sqrt(2)

# Simular ruido (SNR = 15 dB)
snr_db = 15
snr_linear = 10**(snr_db/10)
noise_power = 1.0 / snr_linear
noise = np.random.normal(0, np.sqrt(noise_power/2), size=n_symbols) + \
        1j * np.random.normal(0, np.sqrt(noise_power/2), size=n_symbols)

received_symbols = qpsk_symbols + noise

# Simular oversampling (como en el sistema real)
sps = 16  # samples per symbol
oversampled = np.zeros(n_symbols * sps, dtype=complex)
for i in range(n_symbols):
    # Simular el pulso RRC: transición gradual
    start_idx = i * sps
    end_idx = start_idx + sps
    
    # Transición suave del símbolo anterior al actual
    if i > 0:
        prev_symbol = received_symbols[i-1]
    else:
        prev_symbol = 0
    
    current_symbol = received_symbols[i]
    
    # Interpolación lineal + algo de ruido para simular el pulso
    for j in range(sps):
        alpha = j / sps
        oversampled[start_idx + j] = (1-alpha) * prev_symbol + alpha * current_symbol
        # Agregar pequeño ruido adicional por muestra
        oversampled[start_idx + j] += np.random.normal(0, noise_power*0.1) + \
                                      1j * np.random.normal(0, noise_power*0.1)

# Crear figura comparativa
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 1. Constelación con sobremuestreo (como se ve ahora - líneas)
ax1 = axes[0]
ax1.plot(oversampled.real, oversampled.imag, '.', markersize=0.5, alpha=0.5)
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)
ax1.set_xlabel('In-Phase (I)')
ax1.set_ylabel('Quadrature (Q)')
ax1.set_title(f'Sobremuestreado (sps={sps})\n{len(oversampled)} puntos - SE VEN LÍNEAS')
ax1.set_xlim(-2, 2)
ax1.set_ylim(-2, 2)

# Agregar puntos ideales de referencia
ideal_qpsk = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
ax1.plot(ideal_qpsk.real, ideal_qpsk.imag, 'r*', markersize=15, 
         label='Símbolos ideales', zorder=10)
ax1.legend()

# 2. Constelación con 1 punto por símbolo (símbolos muestreados)
ax2 = axes[1]
ax2.plot(received_symbols.real, received_symbols.imag, '.', markersize=2)
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)
ax2.set_xlabel('In-Phase (I)')
ax2.set_ylabel('Quadrature (Q)')
ax2.set_title(f'1 símbolo muestreado\n{len(received_symbols)} puntos - NUBES DISCRETAS')
ax2.set_xlim(-2, 2)
ax2.set_ylim(-2, 2)

# Agregar puntos ideales de referencia
ax2.plot(ideal_qpsk.real, ideal_qpsk.imag, 'r*', markersize=15, 
         label='Símbolos ideales', zorder=10)
ax2.legend()

# 3. Constelación con decimación (menos puntos para mejor visualización)
ax3 = axes[2]
decimation = 5
decimated = received_symbols[::decimation]
ax3.plot(decimated.real, decimated.imag, 'o', markersize=4)
ax3.set_aspect('equal')
ax3.grid(True, alpha=0.3)
ax3.set_xlabel('In-Phase (I)')
ax3.set_ylabel('Quadrature (Q)')
ax3.set_title(f'Diezmado (1 cada {decimation})\n{len(decimated)} puntos - MÁS CLARO')
ax3.set_xlim(-2, 2)
ax3.set_ylim(-2, 2)

# Agregar puntos ideales de referencia
ax3.plot(ideal_qpsk.real, ideal_qpsk.imag, 'r*', markersize=15, 
         label='Símbolos ideales', zorder=10)
ax3.legend()

plt.suptitle(f'Comparación de visualización de constelaciones QPSK (SNR={snr_db} dB)', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plots/constellation_comparison.png', dpi=150, bbox_inches='tight')
print("Gráfico guardado en plots/constellation_comparison.png")
plt.show()

# Imprimir estadísticas
print("\n" + "="*60)
print("ANÁLISIS DE CONSTELACIONES")
print("="*60)
print(f"Símbolos transmitidos: {n_symbols}")
print(f"Sobremuestreo (sps): {sps}")
print(f"SNR: {snr_db} dB")
print(f"\n1. Sobremuestreado:")
print(f"   - Puntos totales: {len(oversampled)}")
print(f"   - Apariencia: LÍNEAS (muchas muestras por símbolo)")
print(f"   - Ventaja: Muestra la forma del pulso")
print(f"   - Desventaja: Difícil ver la distribución de símbolos")

print(f"\n2. 1 punto por símbolo:")
print(f"   - Puntos totales: {len(received_symbols)}")
print(f"   - Apariencia: NUBES (distribución gaussiana)")
print(f"   - Ventaja: Clara visualización del ruido y SNR")
print(f"   - Desventaja: Puede ser demasiados puntos si N es grande")

print(f"\n3. Diezmado (1 cada {decimation}):")
print(f"   - Puntos totales: {len(decimated)}")
print(f"   - Apariencia: NUBES MÁS CLARAS")
print(f"   - Ventaja: Visualización óptima, menos sobrecarga")
print(f"   - Desventaja: Menor resolución estadística")

print("\n" + "="*60)
print("RECOMENDACIÓN PARA EL SIMULADOR:")
print("="*60)
print("✓ Usar 1 punto por símbolo (ya implementado en chain.py)")
print("✓ Limitar a ~2000 puntos por constelación para renderizado")
print("✓ Esto da visualización clara sin perder información importante")
print("="*60)
