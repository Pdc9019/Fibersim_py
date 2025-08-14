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

    zCum = 0.0
    nextConst = step_const_m
    consZ: list[float] = []
    consSym: list = []
    powZ: list[float] = []
    powW: list[float] = []

    for k, blk in enumerate(chain):
        btype = blk.get("type")
        par = blk.get("par", {})

        if btype == "fiber":
            # Particiona cada tramo en segmentos de step_const_m para capturar después de cada segmento
            Lpend = par["L"]
            parF = dict(par)
            while Lpend > 0:
                Lstep = min(step_const_m, Lpend) if do_const else Lpend
                parF["L"] = Lstep
                A, info = fiber_ssfm(A, info, parF, dz_override=dz_override)
                zCum += Lstep
                Lpend -= Lstep

                # Snapshot
                if do_const:
                    B = rxF(A)
                    delay = 2 * info["pulseDelay"]  # Tx + Rx
                    sym = B[delay::sps]
                    consSym.append(sym)
                    consZ.append(zCum)
                    powZ.append(zCum)
                    powW.append(float(xp.mean(xp.abs(A) ** 2)))

            # pérdida por fusión si el siguiente bloque también es fibra
            if use_splice_loss and k < len(chain) - 1 and chain[k + 1].get("type") == "fiber" and splice_dB:
                A = A * (10 ** (-splice_dB / 20.0))

        elif btype == "edfa":
            A, info = edfa_block(A, info, par)
            # también podemos registrar potencia tras EDFA
            if do_const:
                B = rxF(A)
                delay = 2 * info["pulseDelay"]
                sym = B[delay::sps]
                consSym.append(sym)
                consZ.append(zCum)  # misma distancia, después del EDFA
                powZ.append(zCum)
                powW.append(float(xp.mean(xp.abs(A) ** 2)))
        else:
            raise ValueError(f"Bloque no soportado aún: {btype}")

    Arx = rxF(A)
    info["Lcum"] = zCum

    diag = {
        "consZ_m": consZ,
        "consSym": consSym,
        "powZ_m": powZ,
        "powW_W": powW,
        "delay_samp": 2 * info["pulseDelay"],
        "sps": sps,
    }
    return Arx, info, diag
