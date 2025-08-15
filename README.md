# FiberSim

Simulador ligero de enlaces ópticos en banda base con pulso RRC, propagación SSFM y visualizaciones de constelaciones, eye diagram, evolución de potencia y perfiles z de Potencia, OSNR y BER.  
Incluye GUI en Streamlit y CLI con Typer. Soporta aceleración con GPU usando CuPy.

---

## Tabla de contenidos

- [Características](#características)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Ejecución](#ejecución)
  - [GUI](#gui)
  - [CLI](#cli)
- [Ejemplo de configuración](#ejemplo-de-configuración)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Consejos de rendimiento](#consejos-de-rendimiento)
- [Solución de problemas](#solución-de-problemas)
- [Licencia](#licencia)

---

## Características

- Builder visual con bloques FIBER y EDFA  
  mover, duplicar y eliminar desde mini tarjetas compactas
- Parámetros globales simples y robustos  
  Rb en Gbaud, Ptx en dBm, calidad que fija Nsym, Fs derivado de Rb y sps
- Plots incluidos  
  constelaciones, eye diagram, evolución de potencia, 3D interactivo HTML
- Perfiles z dentro de la GUI  
  Potencia [dBm], OSNR [dB] con Bo configurable y BER BPSK  
  descarga de perfiles a CSV y JSON
- Presets de ejemplo  
  80 km SMF, 3 spans SMF+EDFA, 400 km con DCF
- CPU o GPU sin tocar código  
  conmutación de backend NumPy o CuPy mediante un toggle

---

## Requisitos

- Python 3.10 o superior
- Windows, Linux o macOS

Dependencias principales:
- numpy, scipy, matplotlib, plotly, streamlit
- typer, rich
- pydantic >= 2

Opcional para GPU:
- NVIDIA con CUDA y paquete cupy-cudaXX compatible

Ejemplo de `requirements.txt`:

```txt
numpy>=1.23
scipy>=1.9
matplotlib>=3.7
plotly>=5.17
streamlit>=1.30
typer[all]>=0.9
rich>=13.0
pydantic>=2.4
# GPU opcional, instala solo si tienes CUDA
# cupy-cuda12x>=12.0
# o
# cupy-cuda11x>=12.0
