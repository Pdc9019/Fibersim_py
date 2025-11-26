"""
Módulo para visualización y exportación de waveforms de señales TX vs RX.
Permite comparar señal transmitida vs recibida y exportar en HDF5 para análisis posterior.
"""
from __future__ import annotations
import pathlib
from typing import Any, Dict, Tuple
import numpy as np

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


def save_waveforms_hdf5(
    tx_signal: np.ndarray,
    rx_signal: np.ndarray,
    filepath: str | pathlib.Path,
    metadata: Dict[str, Any] | None = None,
    segment_start: int = 0,
    segment_length: int | None = None,
    rx_signal_post_mf: np.ndarray | None = None
) -> None:
    """
    Guarda waveforms TX y RX en formato HDF5 con metadata.
    
    Args:
        tx_signal: Señal transmitida (compleja)
        rx_signal: Señal recibida (compleja)
        filepath: Ruta del archivo HDF5 a crear
        metadata: Diccionario con metadata (Fs, sps, Ptx, etc.)
        segment_start: Índice de inicio del segmento a guardar
        segment_length: Longitud del segmento (None = todo)
        rx_signal_post_mf: Señal RX después de matched filter (opcional)
    """
    if not HAS_H5PY:
        raise ImportError("h5py no está instalado. Instala con: pip install h5py")
    
    # Convertir a numpy si viene de cupy
    if hasattr(tx_signal, 'get'):
        tx_signal = tx_signal.get()
    if hasattr(rx_signal, 'get'):
        rx_signal = rx_signal.get()
    
    # Extraer segmento
    if segment_length is None:
        segment_length = len(tx_signal) - segment_start
    
    end_idx = min(segment_start + segment_length, len(tx_signal))
    tx_seg = tx_signal[segment_start:end_idx]
    rx_seg = rx_signal[segment_start:end_idx] if len(rx_signal) >= end_idx else rx_signal[segment_start:]
    
    filepath = pathlib.Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(filepath, 'w') as f:
        # Guardar señales complejas como datasets
        f.create_dataset('tx/real', data=tx_seg.real, compression='gzip')
        f.create_dataset('tx/imag', data=tx_seg.imag, compression='gzip')
        f.create_dataset('rx/real', data=rx_seg.real, compression='gzip')
        f.create_dataset('rx/imag', data=rx_seg.imag, compression='gzip')
        
        # Guardar RX post matched filter si está disponible
        if rx_signal_post_mf is not None:
            rx_post_seg = rx_signal_post_mf[segment_start:end_idx] if len(rx_signal_post_mf) >= end_idx else rx_signal_post_mf[segment_start:]
            f.create_dataset('rx_post_mf/real', data=rx_post_seg.real, compression='gzip')
            f.create_dataset('rx_post_mf/imag', data=rx_post_seg.imag, compression='gzip')
        
        # Guardar índices del segmento
        f.attrs['segment_start'] = segment_start
        f.attrs['segment_length'] = len(tx_seg)
        
        # Guardar metadata
        if metadata:
            for key, value in metadata.items():
                try:
                    f.attrs[key] = value
                except (TypeError, ValueError):
                    f.attrs[key] = str(value)


def load_waveforms_hdf5(filepath: str | pathlib.Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Carga waveforms desde archivo HDF5.
    
    Returns:
        (tx_signal, rx_signal, metadata)
    """
    if not HAS_H5PY:
        raise ImportError("h5py no está instalado. Instala con: pip install h5py")
    
    with h5py.File(filepath, 'r') as f:
        tx_real = f['tx/real'][:]
        tx_imag = f['tx/imag'][:]
        rx_real = f['rx/real'][:]
        rx_imag = f['rx/imag'][:]
        
        tx_signal = tx_real + 1j * tx_imag
        rx_signal = rx_real + 1j * rx_imag
        
        metadata = dict(f.attrs)
    
    return tx_signal, rx_signal, metadata


def plot_waveform_comparison(
    tx_signal: np.ndarray,
    rx_signal: np.ndarray,
    sps: int,
    Fs: float,
    segment_start_us: float = 0.0,
    segment_length_us: float = 1.0,
    filepath: str | pathlib.Path | None = None,
    rx_post_dsp: np.ndarray | None = None,
    show_snr_metrics: bool = True
) -> Any:
    """
    Crea gráfico comparativo de waveforms TX vs RX (pre-DSP) vs RX (post-DSP).
    
    Muestra el efecto del sistema óptico y del procesamiento digital:
    - TX: Señal transmitida con pulse shaping + AWGN TX
    - RX (pre-DSP): Señal después de propagación óptica + AWGN RX (antes de matched filter)
    - RX (post-DSP): Señal después de matched filter + equalización (si se provee)
    
    Args:
        tx_signal: Señal transmitida
        rx_signal: Señal recibida pre-DSP (después de propagación óptica)
        sps: Muestras por símbolo
        Fs: Frecuencia de muestreo [Hz]
        segment_start_us: Inicio del segmento a graficar [μs]
        segment_length_us: Duración del segmento [μs]
        filepath: Si se provee, guarda la figura
        rx_post_dsp: Señal recibida post-DSP (después de matched filter, opcional)
        show_snr_metrics: Si True, calcula y muestra SNR en títulos
        
    Returns:
        Figura de matplotlib
    """
    import matplotlib.pyplot as plt
    
    # Convertir a numpy si necesario
    if hasattr(tx_signal, 'get'):
        tx_signal = tx_signal.get()
    if hasattr(rx_signal, 'get'):
        rx_signal = rx_signal.get()
    if rx_post_dsp is not None and hasattr(rx_post_dsp, 'get'):
        rx_post_dsp = rx_post_dsp.get()
    
    # Función helper para calcular SNR de señal (potencia / varianza temporal)
    def calc_signal_snr(signal):
        """Estima SNR asumiendo señal modulada + ruido"""
        power = np.mean(np.abs(signal)**2)
        # Varianza de la señal (incluye modulación + ruido)
        variance = np.var(np.abs(signal))
        # Aproximación: ruido ≈ varianza de la envolvente
        if variance > 0 and power > 0:
            # SNR aproximado (no es exacto sin señal de referencia)
            snr_approx = power / variance
            return 10 * np.log10(snr_approx) if snr_approx > 0 else None
        return None
    
    # Calcular índices del segmento
    dt = 1.0 / Fs  # [s]
    start_idx = int(segment_start_us * 1e-6 / dt)
    length_samples = int(segment_length_us * 1e-6 / dt)
    end_idx = min(start_idx + length_samples, len(tx_signal))
    
    # Extraer segmentos
    tx_seg = tx_signal[start_idx:end_idx]
    rx_seg = rx_signal[start_idx:end_idx] if len(rx_signal) >= end_idx else rx_signal[start_idx:]
    
    # Calcular métricas de potencia y SNR
    tx_power_dbm = 10 * np.log10(np.mean(np.abs(tx_seg)**2) * 1000) if len(tx_seg) > 0 else None
    rx_power_dbm = 10 * np.log10(np.mean(np.abs(rx_seg)**2) * 1000) if len(rx_seg) > 0 else None
    
    # Vector de tiempo en microsegundos
    t_us = np.arange(len(tx_seg)) * dt * 1e6 + segment_start_us
    
    # Determinar número de subplots
    n_plots = 3 if rx_post_dsp is not None else 2
    
    # Crear figura
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 3*n_plots), sharex=True)
    if n_plots == 2:
        axes = [axes[0], axes[1]]
    
    # TX - I y Q en el mismo plot
    axes[0].plot(t_us, tx_seg.real, 'b-', linewidth=1.0, label='I (In-phase)', alpha=0.8)
    axes[0].plot(t_us, tx_seg.imag, 'r-', linewidth=1.0, label='Q (Quadrature)', alpha=0.8)
    axes[0].set_ylabel('Amplitud [a.u.]', fontsize=11)
    
    title_tx = 'Señal Transmitida (TX) - Post Pulse Shaping + AWGN TX'
    if show_snr_metrics and tx_power_dbm is not None:
        title_tx += f'\nPotencia: {tx_power_dbm:.2f} dBm'
    axes[0].set_title(title_tx, fontsize=12, fontweight='bold', pad=10)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].legend(loc='upper right', framealpha=0.9, fontsize=10)
    axes[0].axhline(0, color='k', linewidth=0.5, alpha=0.3)
    
    # RX Pre-DSP - I y Q en el mismo plot
    axes[1].plot(t_us[:len(rx_seg)], rx_seg.real, 'b-', linewidth=1.0, label='I (In-phase)', alpha=0.8)
    axes[1].plot(t_us[:len(rx_seg)], rx_seg.imag, 'r-', linewidth=1.0, label='Q (Quadrature)', alpha=0.8)
    axes[1].set_ylabel('Amplitud [a.u.]', fontsize=11)
    if n_plots == 2:
        axes[1].set_xlabel('Tiempo [μs]', fontsize=11)
    
    title_rx = 'Señal Recibida (RX Pre-DSP) - Post Propagación Óptica + AWGN RX'
    if show_snr_metrics and rx_power_dbm is not None:
        gain_db = rx_power_dbm - tx_power_dbm if tx_power_dbm is not None else None
        title_rx += f'\nPotencia: {rx_power_dbm:.2f} dBm'
        if gain_db is not None:
            title_rx += f' (Ganancia sistema: {gain_db:.2f} dB)'
    axes[1].set_title(title_rx, fontsize=12, fontweight='bold', pad=10)
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].legend(loc='upper right', framealpha=0.9, fontsize=10)
    axes[1].axhline(0, color='k', linewidth=0.5, alpha=0.3)
    
    # RX Post-DSP (si se provee)
    if rx_post_dsp is not None:
        # rx_post_dsp está a tasa de sobremuestreo (después de matched filter, antes de downsampling)
        rx_post_seg = rx_post_dsp[start_idx:end_idx] if len(rx_post_dsp) >= end_idx else rx_post_dsp[start_idx:]
        rx_post_power_dbm = 10 * np.log10(np.mean(np.abs(rx_post_seg)**2) * 1000) if len(rx_post_seg) > 0 else None
        
        axes[2].plot(t_us[:len(rx_post_seg)], rx_post_seg.real, 'b-', linewidth=1.0, 
                    label='I (In-phase)', alpha=0.8)
        axes[2].plot(t_us[:len(rx_post_seg)], rx_post_seg.imag, 'r-', linewidth=1.0, 
                    label='Q (Quadrature)', alpha=0.8)
        axes[2].set_ylabel('Amplitud [a.u.]', fontsize=11)
        axes[2].set_xlabel('Tiempo [μs]', fontsize=11)
        
        title_post = 'Señal Recibida (RX Post-DSP) - Post Matched Filter RRC'
        if show_snr_metrics and rx_post_power_dbm is not None:
            title_post += f'\nPotencia: {rx_post_power_dbm:.2f} dBm'
        axes[2].set_title(title_post, fontsize=12, fontweight='bold', pad=10)
        axes[2].grid(True, alpha=0.3, linestyle='--')
        axes[2].legend(loc='upper right', framealpha=0.9, fontsize=10)
        axes[2].axhline(0, color='k', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    
    if filepath:
        filepath = pathlib.Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filepath, dpi=120, bbox_inches='tight')
    
    return fig
