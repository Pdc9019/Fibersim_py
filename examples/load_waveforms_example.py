"""
Ejemplo de cómo cargar y procesar waveforms guardados desde el simulador.

Los waveforms se guardan en formato HDF5 con:
- Señal TX (transmitida)
- Señal RX (recibida)
- Metadata (Fs, sps, Ptx, modulación, etc.)

Este script muestra cómo:
1. Cargar los waveforms
2. Aplicar procesamiento adicional (ej: compensación)
3. Plotear segmentos personalizados
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Importar módulo de waveforms
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from fibersim.core.waveform import load_waveforms_hdf5, plot_waveform_comparison


def main():
    # Ruta al archivo HDF5 generado por el simulador
    waveform_file = Path("plots/waveforms.h5")
    
    if not waveform_file.exists():
        print(f"Error: No se encontró {waveform_file}")
        print("Ejecuta primero una simulación con 'Waveforms TX/RX' habilitado")
        return
    
    # Cargar waveforms
    print("Cargando waveforms...")
    tx_signal, rx_signal, metadata = load_waveforms_hdf5(waveform_file)
    
    print(f"\nMetadata:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")
    
    print(f"\nSeñales cargadas:")
    print(f"  TX: {len(tx_signal)} muestras")
    print(f"  RX: {len(rx_signal)} muestras")
    
    # Ejemplo 1: Plotear diferentes segmentos
    Fs = float(metadata['Fs'])
    sps = int(metadata['sps'])
    
    # Segmento al inicio
    fig1 = plot_waveform_comparison(
        tx_signal, rx_signal,
        sps=sps, Fs=Fs,
        segment_start_us=0.0,
        segment_length_us=2.0,
        filepath="plots/waveform_inicio.png"
    )
    plt.close(fig1)
    print("\nGráfico guardado: plots/waveform_inicio.png")
    
    # Segmento en el medio
    total_time_us = len(tx_signal) / Fs * 1e6
    mid_time_us = total_time_us / 2
    
    fig2 = plot_waveform_comparison(
        tx_signal, rx_signal,
        sps=sps, Fs=Fs,
        segment_start_us=mid_time_us,
        segment_length_us=2.0,
        filepath="plots/waveform_medio.png"
    )
    plt.close(fig2)
    print("Gráfico guardado: plots/waveform_medio.png")
    
    # Ejemplo 2: Análisis de potencia
    tx_power_avg = np.mean(np.abs(tx_signal)**2)
    rx_power_avg = np.mean(np.abs(rx_signal)**2)
    attenuation_dB = 10 * np.log10(rx_power_avg / tx_power_avg)
    
    print(f"\nAnálisis de potencia:")
    print(f"  Potencia TX promedio: {10*np.log10(tx_power_avg):.2f} dBW")
    print(f"  Potencia RX promedio: {10*np.log10(rx_power_avg):.2f} dBW")
    print(f"  Atenuación total: {attenuation_dB:.2f} dB")
    
    # Ejemplo 3: Espectro de frecuencias
    print("\nCalculando espectro...")
    tx_fft = np.fft.fftshift(np.fft.fft(tx_signal))
    rx_fft = np.fft.fftshift(np.fft.fft(rx_signal))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(tx_signal), 1/Fs)) / 1e9  # GHz
    
    fig3, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    axes[0].plot(freqs, 10*np.log10(np.abs(tx_fft)**2 + 1e-12), 'b-', linewidth=0.8)
    axes[0].set_ylabel('PSD TX [dB]')
    axes[0].set_title('Densidad Espectral de Potencia')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([-Fs/2e9, Fs/2e9])
    
    axes[1].plot(freqs, 10*np.log10(np.abs(rx_fft)**2 + 1e-12), 'r-', linewidth=0.8)
    axes[1].set_ylabel('PSD RX [dB]')
    axes[1].set_xlabel('Frecuencia [GHz]')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([-Fs/2e9, Fs/2e9])
    
    plt.tight_layout()
    plt.savefig("plots/waveform_spectrum.png", dpi=150)
    plt.close(fig3)
    print("Gráfico guardado: plots/waveform_spectrum.png")
    
    print("\n¡Listo! Revisa la carpeta 'plots/' para ver los resultados.")


if __name__ == "__main__":
    main()
