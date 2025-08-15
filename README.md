# FiberSim

Simulador ligero de enlaces ópticos con shaping RRC, propagación SSFM y visualizaciones de constelaciones, eye diagram, evolución de potencia y perfiles z de Potencia, OSNR y BER.  
Incluye GUI en Streamlit y CLI con Typer. Soporta CPU con NumPy y, opcionalmente, GPU con CuPy.

---

## Características

- Builder visual por bloques: FIBER y EDFA. Mover, duplicar y eliminar desde mini tarjetas.
- Parámetros globales simples y robustos: Rb en Gbaud, Ptx en dBm, calidad que fija Nsym y Fs derivado.
- Plots: constelaciones, eye diagram, potencia vs z, 3D interactivo HTML de constelaciones.
- Perfiles z en la GUI: Potencia [dBm], OSNR [dB] con Bo configurable y BER BPSK. Exportar a CSV y JSON.
- Presets de ejemplo listos para cargar.
- Conmutación rápida CPU o GPU mediante un toggle en la GUI o flag en la CLI.

---

## Requisitos

- Python 3.10 o superior
- Windows, Linux o macOS

Dependencias principales:
- numpy, scipy, matplotlib, plotly, streamlit
- typer, rich
- pydantic 2.x

Opcional para GPU:
- cupy-cudaXX compatible con tu versión de CUDA

Archivo sugerido `requirements.txt`:

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
```

---

## Instalación

Clona el repo y crea un entorno:

```bash
git clone https://github.com/tu-usuario/FiberSim.git
cd FiberSim

# venv de ejemplo
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

GPU opcional con CuPy: instala el wheel que corresponda a tu CUDA.

```bash
# Ejemplos, elige uno
pip install cupy-cuda12x
# o
pip install cupy-cuda11x
```

---

## Ejecución de la GUI

```bash
streamlit run src/fibersim/gui/app.py
```

Características de la GUI:
- Builder por bloques con mini tarjetas y acciones rápidas.
- Parámetros simples y robustos: Rb en Gbaud, Ptx en dBm, calidad predefinida.
- Toggle CPU o GPU.
- Descarga y carga de configuraciones JSON. Presets incluidos.
- Resultados: constelaciones, eye diagram, potencia vs z y 3D interactivo.
- Panel de métricas: backend, tiempo, OSNR, Pout y BER estimado.
- Perfiles z analíticos desde la propia GUI. Exportables a CSV y JSON.

---

## Ejecución por CLI

Ejecuta el simulador desde consola con Typer:

```bash
# Si tu entorno no está instalado como paquete, exporta PYTHONPATH
# Windows (PowerShell)
$env:PYTHONPATH = "."
# Linux/macOS
# export PYTHONPATH=.

python -m fibersim.main run --config tmp_gui_config.json --gpu true --plots-dir plots --outdir logs
```

Parámetros principales:
- `--gpu [true|false]` activa o desactiva CuPy.
- `--dz` override del paso SSFM.
- `--do-const` captura de constelaciones intermedias.
- `--step-const-km` distancia entre snapshots de constelaciones.
- `--do-eye` guarda eye diagram final.
- `--plots-dir` y `--outdir` para salidas.

---

## Estructura del repo

```
src/
  fibersim/
    core/           # Núcleo numérico: PRBS, shaping, SSFM, EDFA, plots
    gui/            # App Streamlit (app.py)
    main.py         # CLI con Typer y orquestación
    schema.py       # Pydantic models para validación de config
    io.py           # Logs JSON
plots/              # Imágenes y HTML generados
logs/               # Logs JSON de simulación
```

---

## Presets incluidos

- Demo 400 km con DCF y EDFAs intermedios.
- Demo 80 km solo SMF.
- Demo 3 spans SMF + EDFA.

Se cargan desde la barra lateral de la GUI.

---

## Notas de rendimiento

- Con GPU: el FFT y SSFM suelen acelerar significativamente con CuPy.
- La GUI permite limitar puntos del 3D HTML para mantener fluidez.
- Ajusta `Nsym` con la calidad de simulación para probar rápido o con más fidelidad.

---

## Solución de problemas

- BER ~0.5 con enlace corto: verifica delay total TX+RX para el slice de símbolos. La app alinea automáticamente, pero si modificas el pipeline revisa `delay_samp`.
- OSNR final n/a: se reporta cuando no hay bloques EDFA en la cadena o no se acumuló ruido ASE en el perfil. Agrega un EDFA con `nsp` válido.
- CPU/GPU siempre igual: asegúrate de que el toggle esté activo y que CuPy esté instalado. La CLI muestra el backend elegido.
- ImportError ejecutando la CLI: exporta `PYTHONPATH=.` o instala el paquete con `pip install -e .`.

---

## Roadmap corto

- EDFA con saturación simple y NF dependiente de ganancia.
- Barridos Ptx y LDCF con heatmaps.
- Modos QPSK y 16QAM con receptor coherente básico.
- GN model para NLI en WDM como estimación rápida.
- Más métricas y plots acoplados a la GUI.

---


