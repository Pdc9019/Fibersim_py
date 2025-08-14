from __future__ import annotations
from typing import Literal, List, Union
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

class SimConfig(BaseModel):
    global_: GlobalPar = Field(..., alias="global")
    pulse: PulsePar
    chain: List[Block]

    class Config:
        populate_by_name = True
