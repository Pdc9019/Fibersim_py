from __future__ import annotations
import math
from typing import Dict

def gn_snr_per_channel(
    Nspan: int,
    alpha_1_per_km: float,
    beta2_ps2_per_km: float,
    gamma_W_1_per_km: float,
    Rs_Gbaud: float,
    Pch_dBm: float,
    Nch: int,
    delta_f_GHz: float,
    NF_dB: float,
    lambda_nm: float = 1550.0,
) -> Dict[str, float]:
    """
    GN-model simplificado: devuelve SNR_NLI, OSNR_ASE y SNR_total estimados.
    Referencia: Poggiolini, formulas estándar, con constantes empaquetadas.
    """
    # unidades
    alpha = alpha_1_per_km * 1e-3
    beta2 = beta2_ps2_per_km * 1e-24
    gamma = gamma_W_1_per_km * 1e-3
    Rs = Rs_Gbaud * 1e9
    df = delta_f_GHz * 1e9
    Pch = 1e-3 * 10**(Pch_dBm/10)

    # L_eff
    Leff = (1 - math.exp(-2*alpha*1e3)) / (2*alpha)  # aproximación por span corto 1 km, ajusta si quieres por L_span
    # ruído ASE por span en banda equivalente a Rs
    from .edfa import dB_to_lin
    h = 6.62607015e-34
    c = 299792458.0
    nu = c / (lambda_nm*1e-9)
    NF = dB_to_lin(NF_dB)
    G_lin = dB_to_lin(20.0)  # ejemplo, ajusta si sabes la ganancia real
    Pase_span = (NF/2.0) * 2*h*nu*(G_lin-1)*Rs

    # NLI PSD aprox (coeficiencia eta)
    # eta ~ (16/27) * gamma^2 / (2*pi*abs(beta2) ) * log(pi^2*abs(beta2)*Nch^2*Rs^2/(2*alpha))
    try:
        eta = (16.0/27.0) * (gamma**2) / (2*math.pi*abs(beta2)) * math.log( (math.pi**2)*abs(beta2)*(Nch**2)*(Rs**2)/(2*alpha) )
        Pnli_span = eta * (Pch**3) * Rs
    except Exception:
        eta = 0.0
        Pnli_span = 0.0

    P_ASE_total = Nspan * Pase_span
    P_NLI_total = Nspan * Pnli_span

    SNR_ASE = (Pch) / max(P_ASE_total, 1e-30)
    SNR_NLI = (Pch) / max(P_NLI_total, 1e-30)
    SNR_tot = 1.0 / (1.0/SNR_ASE + 1.0/SNR_NLI)

    def to_dB(x): return 10*math.log10(max(x, 1e-30))
    return dict(SNR_ASE_dB=to_dB(SNR_ASE), SNR_NLI_dB=to_dB(SNR_NLI), SNR_total_dB=to_dB(SNR_tot))
