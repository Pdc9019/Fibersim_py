from __future__ import annotations
from typing import Literal, List, Union, Optional
from pydantic import BaseModel, Field, PositiveFloat, NonNegativeFloat, ValidationError

# ---- Modelos de bloques ----
class FiberPar(BaseModel):
    L: PositiveFloat = Field(..., description="Longitud de tramo (m)")
    beta2: float
    gamma: PositiveFloat
    dz: PositiveFloat
    alpha: NonNegativeFloat  # atenuación [1/m]

class EdfaPar(BaseModel):
    G_dB: float
    nsp: PositiveFloat

class FiberBlock(BaseModel):
    type: Literal["fiber"]
    par: FiberPar

class EdfaBlock(BaseModel):
    type: Literal["edfa"]
    par: EdfaPar

Block = Union[FiberBlock, EdfaBlock]

# ---- Secciones globales ----
class PulsePar(BaseModel):
    type: Literal["RRC"] = "RRC"
    roll: PositiveFloat
    span: int = Field(..., ge=1)

class GlobalPar(BaseModel):
    Rb: PositiveFloat
    M: int = Field(..., ge=2)
    sps: int = Field(..., ge=2)
    Fs: PositiveFloat
    Nsym: int = Field(..., ge=1)
    Ptx: PositiveFloat
    # Nuevos parámetros globales (backwards-compatible con defaults)
    mod: Literal["BPSK", "QPSK", "16QAM"] = "BPSK"
    rx: Literal["imdd", "coh"] = "imdd"
    pol: Literal["sp", "dp"] = "sp"
    
    # Ruido AWGN (Additive White Gaussian Noise) - Unified control
    enable_awgn: bool = False  # Activar ruido gaussiano en TX y RX
    awgn_intensity_db: float = 25.0  # Intensidad del ruido (SNR base) [dB]

class DspPar(BaseModel):
    """Parámetros de DSP en recepción coherente.

    Todos con valores por defecto para mantener compatibilidad.
    """
    timing_algo: Literal["none", "mm"] = "mm"
    eq_taps: int = 11
    eq_mu: float = 1e-3
    phase_algo: Literal["none", "bps", "vv"] = "bps"

class SimConfig(BaseModel):
    global_: GlobalPar = Field(..., alias="global")
    pulse: PulsePar
    chain: List[Block]
    dsp: Optional[DspPar] = None

    class Config:
        populate_by_name = True
