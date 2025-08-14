from __future__ import annotations
from typing import Tuple
import math

h = 6.62607015e-34
c = 299792458.0

def dB_to_lin(x_dB: float) -> float:
    return 10.0**(x_dB/10.0)

def lin_to_dB(x: float) -> float:
    return 10.0*math.log10(max(x, 1e-30))

def dBm_to_W(p_dBm: float) -> float:
    return 1e-3 * 10.0**(p_dBm/10.0)

def W_to_dBm(p_W: float) -> float:
    return 10.0*math.log10(max(p_W, 1e-30)/1e-3)

def nsp_from_nf_dB(nf_dB: float) -> float:
    # NF_lin = 2*nsp en EDFA ideal de banda ancha
    NF_lin = dB_to_lin(nf_dB)
    return 0.5*NF_lin

def nf_dB_vs_gain(G_dB: float, nf_min_dB: float = 4.5, G_ref_dB: float = 20.0, slope_dB_per_dB: float = 0.03) -> float:
    """
    NF que crece suave cuando te alejas de la ganancia de diseño.
    """
    delta = abs(G_dB - G_ref_dB)
    return nf_min_dB + slope_dB_per_dB * delta

def sat_gain_dB(Pin_W: float, G0_dB: float, Psat_out_dBm: float) -> float:
    """
    Ganancia efectiva por saturación de salida: G_eff = G0 / (1 + Pout/Psat).
    """
    G0 = dB_to_lin(G0_dB)
    Psat_W = dBm_to_W(Psat_out_dBm)
    Pout_lin = G0 * Pin_W
    G_eff_lin = G0 / (1.0 + max(Pout_lin, 0.0)/max(Psat_W, 1e-12))
    return lin_to_dB(G_eff_lin)

def ase_power_W(G_dB: float, nf_dB: float, lambda_nm: float, Bo_Hz: float) -> float:
    """
    P_ASE de doble polarización en banda óptica Bo.
    """
    nu = c / (lambda_nm*1e-9)
    nsp = nsp_from_nf_dB(nf_dB)
    G = dB_to_lin(G_dB)
    return 2.0 * nsp * h * nu * (G - 1.0) * Bo_Hz
