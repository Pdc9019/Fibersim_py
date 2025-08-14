from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
import math

from .edfa import dB_to_lin, lin_to_dB, W_to_dBm, ase_power_W, nf_dB_vs_gain, sat_gain_dB

def alpha_lin_to_db_per_km(alpha_1_per_m: float) -> float:
    # potencia dB/km
    return 4.343 * alpha_1_per_m * 1e3

def build_power_osnr_profile(
    cfg: Dict[str, Any],
    lambda_nm: float = 1550.0,
    Bo_Hz: float = 12.5e9,
    use_edfa_saturation: bool = True,
    edfa_psat_out_dBm: float = 15.0,
) -> List[Dict[str, Any]]:
    """
    Recorre chain y calcula Potencia y OSNR por bloque con un modelo analítico rápido.
    - Señal: aplica pérdida de fibra y ganancia de EDFA.
    - Ruido: acumula ASE en edfas.
    """
    g = cfg["global"]
    chain = cfg["chain"]

    z_m = 0.0
    # potencia de señal en W
    P_sig_W = float(g["Ptx"])
    # ruido acumulado en W
    P_ase_W = 0.0

    prof: List[Dict[str, Any]] = []
    for i, blk in enumerate(chain):
        t = blk["type"]
        par = blk["par"]

        if t == "fiber":
            L = float(par["L"])
            alpha = float(par["alpha"])  # 1/m
            # atenuación lineal de potencia
            P_sig_W *= math.exp(-alpha * L)
            z_m += L
            prof.append({
                "i": i, "kind": "fiber", "z_km": z_m/1e3,
                "P_dBm": W_to_dBm(P_sig_W),
                "OSNR_dB": None if P_ase_W <= 0 else lin_to_dB(P_sig_W / P_ase_W),
            })

        elif t == "edfa":
            G0_dB = float(par.get("G_dB", 0.0))
            # NF variable simple
            nf_dB = float(par.get("nf_dB", None)) if par.get("nf_dB", None) is not None else nf_dB_vs_gain(G0_dB)
            # Saturación opcional
            if use_edfa_saturation:
                G_eff_dB = sat_gain_dB(P_sig_W, G0_dB, edfa_psat_out_dBm)
            else:
                G_eff_dB = G0_dB

            # amplifica señal
            P_sig_W *= dB_to_lin(G_eff_dB)
            # añade ASE
            P_ase_W += ase_power_W(G_eff_dB, nf_dB, lambda_nm, Bo_Hz)

            prof.append({
                "i": i, "kind": "edfa", "z_km": z_m/1e3,
                "G_dB": G_eff_dB, "NF_dB": nf_dB,
                "P_dBm": W_to_dBm(P_sig_W),
                "OSNR_dB": lin_to_dB(P_sig_W / max(P_ase_W, 1e-30)),
            })
        else:
            # bloque desconocido
            prof.append({
                "i": i, "kind": t, "z_km": z_m/1e3,
                "P_dBm": W_to_dBm(P_sig_W),
                "OSNR_dB": None if P_ase_W <= 0 else lin_to_dB(P_sig_W / P_ase_W),
            })

    return prof

def add_q_ber_to_profile(
    prof: List[Dict[str, Any]],
    M: int,
    Rb: float,
    Bo_Hz: float = 12.5e9,
) -> None:
    """
    Añade Q y BER a los puntos donde hay OSNR disponible.
    Aproximaciones:
      SNR_elec ≈ OSNR_lin * (Bo / Rb)
      BPSK: BER ≈ 0.5*erfc( sqrt(SNR)/sqrt(2) )
      QPSK: BER ≈ 0.5*erfc( sqrt(SNR/2) )
      16QAM: BER ≈ 0.75/4 * erfc( sqrt(0.1*SNR) )  (aprox)
    """
    from math import erfc, sqrt

    for pt in prof:
        OSNR_dB = pt.get("OSNR_dB", None)
        if OSNR_dB is None:
            pt["Q"] = None
            pt["BER"] = None
            continue
        OSNR_lin = 10.0**(OSNR_dB/10.0)
        SNR_lin = OSNR_lin * (Bo_Hz / max(Rb, 1.0))  # mapea ruido óptico a banda de símbolo

        if M == 2:  # BPSK
            Q = max(sqrt(SNR_lin), 1e-12)
            BER = 0.5 * erfc(Q / sqrt(2.0))
        elif M == 4:  # QPSK
            BER = 0.5 * erfc(math.sqrt(0.5*max(SNR_lin, 1e-12)))
            Q = None
        elif M == 16:  # 16QAM
            BER = (0.75/4.0) * erfc(math.sqrt(0.1*max(SNR_lin, 1e-12)))
            Q = None
        else:
            # genérico alto orden, aproximación muy burda
            BER = 0.2 * erfc(math.sqrt(0.1*max(SNR_lin, 1e-12)))
            Q = None

        pt["Q"] = Q
        pt["BER"] = BER
