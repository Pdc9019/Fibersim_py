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
    consZ: List[float] = []
    consSym: List[Any] = []
    powZ: List[float] = []
    powW: List[float] = []

    # --- SNAPSHOT INICIAL EN Z = 0 KM ---
    if do_const:
        B0 = rxF(A)
        delay0 = 2 * info["pulseDelay"]
        sym0 = B0[delay0::sps]
        consSym.append(sym0)
        consZ.append(0.0)
        powZ.append(0.0)
        powW.append(float(xp.mean(xp.abs(A) ** 2)))

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
                    delay = 2 * info["pulseDelay"]
                    sym = B[delay::sps]
                    consSym.append(sym)
                    consZ.append(zCum)
                    powZ.append(zCum)
                    powW.append(float(xp.mean(xp.abs(A) ** 2)))

            if use_splice_loss and k < len(chain) - 1 and chain[k + 1].get("type") == "fiber" and splice_dB:
                A = A * (10 ** (-splice_dB / 20.0))

        elif btype == "edfa":
            A, info = edfa_block(A, info, par)
            if do_const:
                B = rxF(A)
                delay = 2 * info["pulseDelay"]
                sym = B[delay::sps]
                consSym.append(sym)
                consZ.append(zCum)
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
