from __future__ import annotations
from typing import Dict, Any, List, Tuple, Callable
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from fibersim.analysis.profile import build_power_osnr_profile, add_q_ber_to_profile

def load_cfg(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def sweep_ptx(cfg_path: str, ptx_dBm_grid: List[float], M: int, Rb: float, out_png: str) -> None:
    cfg = load_cfg(cfg_path)
    Z = []
    for p_dBm in ptx_dBm_grid:
        cfg["global"]["Ptx"] = 1e-3*10**(p_dBm/10)
        prof = build_power_osnr_profile(cfg)
        add_q_ber_to_profile(prof, M=M, Rb=Rb)
        Z.append(prof[-1].get("OSNR_dB", None))
    Z = np.array(Z, dtype=float)
    fig = plt.figure(figsize=(6,4))
    plt.plot(ptx_dBm_grid, Z, marker="o")
    plt.xlabel("Ptx [dBm]")
    plt.ylabel("OSNR final [dB]")
    plt.grid(True, alpha=0.3)
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)

def sweep_ldcf(cfg_path: str, fiber_index: int, ldcf_km_grid: List[float], out_png: str) -> None:
    """
    Ajusta L del bloque de fibra dado por índice y evalúa OSNR final analítico.
    Útil si ese bloque representa tu DCF.
    """
    cfg = load_cfg(cfg_path)
    Z = []
    for Lkm in ldcf_km_grid:
        cfg["chain"][fiber_index]["par"]["L"] = float(Lkm*1e3)
        prof = build_power_osnr_profile(cfg)
        Z.append(prof[-1].get("OSNR_dB", None))
    Z = np.array(Z, dtype=float)
    fig = plt.figure(figsize=(6,4))
    plt.plot(ldcf_km_grid, Z, marker="o")
    plt.xlabel("L DCF [km]")
    plt.ylabel("OSNR final [dB]")
    plt.grid(True, alpha=0.3)
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
