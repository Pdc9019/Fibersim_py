import os

USE_GPU_ENV = os.getenv("FIBERSIM_GPU", "1") == "1"

try:
    if USE_GPU_ENV:
        import cupy as xp  # type: ignore
        from cupyx.scipy import signal as xsignal  # type: ignore
    else:
        raise ImportError
except Exception:
    import numpy as xp  # type: ignore
    from scipy import signal as xsignal  # type: ignore

def asnumpy(x):
    """Convierte a numpy si viene de CuPy; si ya es numpy, lo deja igual."""
    try:
        import cupy as _cp  # type: ignore
        if isinstance(x, _cp.ndarray):
            return _cp.asnumpy(x)
    except Exception:
        pass
    return x
