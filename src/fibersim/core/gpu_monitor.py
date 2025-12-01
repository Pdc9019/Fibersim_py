# src/fibersim/core/gpu_monitor.py
"""
Monitoreo simplificado de memoria GPU para simulaciones con CuPy.
Solo usa CuPy nativo sin dependencias adicionales.
"""

from __future__ import annotations
from typing import Dict


class GPUMonitor:
    """Monitor simplificado de memoria GPU para CuPy."""
    
    def __init__(self):
        self._available = False
        self._cp = None
        self._device_id = 0
        
        try:
            import cupy as cp
            self._cp = cp
            self._available = True
            self._device_id = cp.cuda.Device().id
        except (ImportError, Exception):
            pass
    
    def is_available(self) -> bool:
        """Verifica si hay GPU disponible."""
        return self._available
    
    def get_memory_info(self) -> Dict[str, float]:
        """
        Obtiene información de memoria de la GPU.
        
        Returns:
            Dict con:
            - total_gb: Memoria total en GB
            - used_gb: Memoria usada en GB
            - free_gb: Memoria libre en GB
            - used_percent: Porcentaje de uso (0-100)
        """
        if not self._available:
            return {
                "total_gb": 0.0,
                "used_gb": 0.0,
                "free_gb": 0.0,
                "used_percent": 0.0
            }
        
        try:
            # Obtener memoria del dispositivo
            device = self._cp.cuda.Device(self._device_id)
            free_bytes, total_bytes = device.mem_info
            used_bytes = total_bytes - free_bytes
            
            # Convertir a GB
            total_gb = total_bytes / (1024**3)
            used_gb = used_bytes / (1024**3)
            free_gb = free_bytes / (1024**3)
            used_percent = (used_gb / total_gb * 100) if total_gb > 0 else 0.0
            
            return {
                "total_gb": total_gb,
                "used_gb": used_gb,
                "free_gb": free_gb,
                "used_percent": used_percent
            }
        except Exception:
            return {
                "total_gb": 0.0,
                "used_gb": 0.0,
                "free_gb": 0.0,
                "used_percent": 0.0
            }


# Instancia global singleton
_gpu_monitor = None


def get_gpu_monitor() -> GPUMonitor:
    """Obtiene la instancia singleton del monitor de GPU."""
    global _gpu_monitor
    if _gpu_monitor is None:
        _gpu_monitor = GPUMonitor()
    return _gpu_monitor
