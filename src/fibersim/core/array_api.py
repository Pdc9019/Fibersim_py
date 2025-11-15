# src/fibersim/core/array_api.py
from __future__ import annotations
import os
import importlib
from typing import Any

# Variables globales de backend
xp: Any = None
xsignal: Any = None
backend_name: str = "numpy"

def set_backend(use_gpu: bool = False) -> str:
    """Configura el backend de arrays CPU/GPU."""
    global xp, xsignal, backend_name
    
    if use_gpu:
        try:
            import cupy as cp
            import cupyx.scipy.signal as cupy_signal
            
            # Verificar que CuPy funcione
            test_array = cp.array([1.0])
            _ = cp.sqrt(test_array)  # Test básico
            
            xp = cp
            xsignal = cupy_signal
            backend_name = "cupy"
            return "GPU (CuPy)"
        except (ImportError, Exception) as e:
            print(f"Warning: CuPy no disponible, usando CPU. Error: {e}")
            # Fallback a NumPy
    
    # Backend CPU (NumPy)
    import numpy as np
    import scipy.signal as np_signal
    
    xp = np
    xsignal = np_signal
    backend_name = "numpy"
    return "CPU (NumPy)"

def to_backend(array, target_backend=None):
    """Convierte array al backend activo."""
    if target_backend is None:
        target_backend = xp
    
    if backend_name == "cupy" and hasattr(array, "get"):
        # Ya es CuPy
        return array
    elif backend_name == "cupy" and not hasattr(array, "get"):
        # NumPy -> CuPy
        import cupy as cp
        return cp.asarray(array)
    elif backend_name == "numpy" and hasattr(array, "get"):
        # CuPy -> NumPy
        return array.get()
    else:
        # Ya es NumPy o conversión NumPy->NumPy
        import numpy as np
        return np.asarray(array)

def ensure_compatible_arrays(*arrays):
    """Asegura que todos los arrays sean del mismo backend."""
    converted = []
    for arr in arrays:
        converted.append(to_backend(arr))
    return converted if len(converted) > 1 else converted[0]

def asnumpy(x):
    """Convierte a numpy si viene de CuPy; si ya es numpy, lo deja igual."""
    try:
        import cupy as _cp  # type: ignore
        if isinstance(x, _cp.ndarray):
            return _cp.asnumpy(x)
    except Exception:
        pass
    return x

def _setup_backend():
    """Setup automático del backend según FIBERSIM_GPU."""
    use_gpu = os.environ.get("FIBERSIM_GPU", "0") == "1"
    return set_backend(use_gpu)

# Inicializar al importar
_setup_backend()
