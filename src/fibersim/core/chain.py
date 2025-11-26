from __future__ import annotations
from typing import Any, Dict, List, Tuple
from .array_api import xp
from .utils import getfield_def, get_rx_filter
from .fiber import fiber_ssfm
from .edfa import edfa_block

def run_chain(
    Ein,
    info: Dict[str, Any],
    chain: List[Dict[str, Any]],
    parGlob: Dict[str, Any],
    *,
    dz_override: float | None = None,
    use_insertion_loss: bool = True,
    insertion_dB: float = 0.0,
    use_splice_loss: bool = True,
    splice_dB: float = 0.0,
    do_const: bool = True,
    step_const_m: float = 5_000.0,
) -> Tuple["xp.ndarray", Dict[str, Any], Dict[str, Any]]:
    A = Ein
    if use_insertion_loss and insertion_dB:
        A = A * (10 ** (-insertion_dB / 20.0))

    sps = int(parGlob["sps"])
    roll = getfield_def(parGlob, "roll", 0.1)
    span = getfield_def(parGlob, "span", 10)
    rxF = get_rx_filter(sps, roll, span)
    
    # Group delay del filtro RX (muestras)
    rx_filter_delay = (span * sps) // 2

    zCum = 0.0
    consZ: List[float] = []
    consSym: List[Any] = []
    powZ: List[float] = []
    powW: List[float] = []
    osnrZ: List[float] = []  # OSNR en cada punto
    
    # Parámetros para cálculo de OSNR
    Rb = float(parGlob.get("Rb", 32e9))  # Tasa de símbolos (Hz)
    
    # --- Imagen INICIAL EN Z = 0 KM ---
    if do_const:
        B0 = rxF(A)
        # Delay total = TX delay + RX delay
        delay0 = info["pulseDelay"] + rx_filter_delay
        sym0 = B0[delay0::sps]
        consSym.append(sym0)
        consZ.append(0.0)
        powZ.append(0.0)
        P_sig = float(xp.mean(xp.abs(A) ** 2))
        powW.append(P_sig)
        
        # OSNR inicial (antes de amplificadores, infinito en principio)
        P_ase = info.get("P_ASE_total", 0.0)
        if P_ase > 1e-30:
            osnr_lin = P_sig / P_ase
            osnr_db = 10.0 * xp.log10(osnr_lin) if hasattr(xp, 'log10') else 10.0 * float(xp.log(osnr_lin) / xp.log(10.0))
            osnrZ.append(float(osnr_db))
        else:
            osnrZ.append(None)  # Sin ruido ASE aún

    for k, blk in enumerate(chain):
        btype = blk.get("type")
        par = blk.get("par", {})

        if btype == "fiber":
            Lpend = par["L"]
            parF = dict(par)
            while Lpend > 0:
                step = min(step_const_m, Lpend) if do_const else Lpend
                parF["L"] = step
                A, info = fiber_ssfm(A, info, parF, dz_override=dz_override)
                zCum += step
                Lpend -= step

                if do_const:
                    B = rxF(A)
                    # Delay total = TX delay + RX delay
                    delay = info["pulseDelay"] + rx_filter_delay
                    sym = B[delay::sps]
                    consSym.append(sym)
                    consZ.append(zCum)
                    powZ.append(zCum)
                    P_sig = float(xp.mean(xp.abs(A) ** 2))
                    powW.append(P_sig)
                    
                    # Calcular OSNR óptico (solo ASE, no incluye AWGN TX)
                    # El AWGN TX es un ruido eléctrico que viaja con la señal
                    # y se amplifica/atenúa junto con ella, no es ruido óptico
                    P_ase = info.get("P_ASE_total", 0.0)
                    if P_ase > 1e-30:
                        osnr_lin = P_sig / P_ase
                        osnr_db = 10.0 * xp.log10(osnr_lin) if hasattr(xp, 'log10') else 10.0 * float(xp.log(osnr_lin) / xp.log(10.0))
                        osnrZ.append(float(osnr_db))
                    else:
                        osnrZ.append(None)

            if use_splice_loss and k < len(chain) - 1 and chain[k + 1].get("type") == "fiber" and splice_dB:
                A = A * (10 ** (-splice_dB / 20.0))

        elif btype == "edfa":
            A, info = edfa_block(A, info, par)
            if do_const:
                B = rxF(A)
                # Delay total = TX delay + RX delay
                delay = info["pulseDelay"] + rx_filter_delay
                sym = B[delay::sps]
                consSym.append(sym)
                consZ.append(zCum)
                powZ.append(zCum)
                P_sig = float(xp.mean(xp.abs(A) ** 2))
                powW.append(P_sig)
                
                # Calcular OSNR óptico (solo ASE, no incluye AWGN TX)
                # El AWGN TX es un ruido eléctrico que viaja con la señal
                # y se amplifica/atenúa junto con ella, no es ruido óptico
                P_ase = info.get("P_ASE_total", 0.0)
                if P_ase > 1e-30:
                    osnr_lin = P_sig / P_ase
                    osnr_db = 10.0 * xp.log10(osnr_lin) if hasattr(xp, 'log10') else 10.0 * float(xp.log(osnr_lin) / xp.log(10.0))
                    osnrZ.append(float(osnr_db))
                else:
                    osnrZ.append(None)
        else:
            raise ValueError(f"Bloque no soportado aún: {btype}")

    Arx = rxF(A)
    info["Lcum"] = zCum

    # Calcular delay total correctamente: TX filter delay + RX filter delay
    # Cada filtro RRC tiene group delay = span * sps / 2
    # info["pulseDelay"] = span_TX * sps / 2 (guardado del TX)
    # rx_filter_delay = span_RX * sps / 2 (calculado arriba)
    delay_total_samp = info["pulseDelay"] + rx_filter_delay

    diag = {
        "consZ_m": consZ,
        "consSym": consSym,
        "powZ_m": powZ,
        "powW_W": powW,
        "osnrZ_dB": osnrZ,
        "delay_samp": delay_total_samp,
        "sps": sps,
    }
    return Arx, info, diag
