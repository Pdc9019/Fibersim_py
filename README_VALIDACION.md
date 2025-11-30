# 📊 VALIDACIÓN Y EVALUACIÓN DE DESEMPEÑO - FIBERSIM_PY
## Sistema de Validación Completo para Trabajo de Título

---

## 🎯 OBJETIVO

Este sistema proporciona una metodología completa para validar el simulador Fibersim_py, midiendo:
- ✅ Desempeño computacional (speedup GPU vs CPU)
- ✅ Exactitud numérica (convergencia SSFM)
- ✅ Validación física (BER, OSNR, curvas teóricas)
- ✅ Comparación con MATLAB (benchmark 250 km)
- ✅ Escalabilidad (temporal y espacial)

---

## 📁 ARCHIVOS GENERADOS

### Configuraciones de Validación (8 archivos JSON)
Ubicación: `examples/configs/val_*.json`

1. **val_qpsk_40km_baseline.json** - QPSK baseline para análisis de sensibilidad a Ptx
2. **val_qpsk_120km_3spans.json** - Acumulación de ruido ASE en 3 spans
3. **val_16qam_80km_baseline.json** - 16-QAM baseline para no-linealidad
4. **val_16qam_160km_compensated.json** - 160 km con compensación DCF
5. **val_64qam_40km_highsnr.json** - 64-QAM alto orden
6. **val_qpsk_stepsize_comparison.json** - Convergencia numérica vs dz
7. **val_scaling_nsym.json** - Escalabilidad vs número de símbolos
8. **val_benchmark_250km_matlab.json** - ⭐ **Comparación directa con MATLAB**

### Documentos de Planificación

1. **PLAN_VALIDACION.md** - Plan maestro completo
   - 137 simulaciones programadas
   - Matriz de pruebas detallada
   - Métricas a recolectar
   - Procedimientos de ejecución

2. **GUIA_INTEGRACION_LATEX.md** - Guía para integrar en memoria
   - Estructura de secciones
   - 7 tablas LaTeX con plantillas
   - 7 figuras con especificaciones
   - Texto narrativo propuesto

3. **README_VALIDACION.md** - Este archivo

### Scripts de Automatización

1. **run_validation_batch.py** - Ejecuta todas las validaciones automáticamente
   - Detecta plataforma (RTX 3060 / RTX 4060)
   - Itera sobre configuraciones y variaciones
   - Guarda resultados en CSV
   - Logging estructurado

2. **process_validation_results.py** - Procesa resultados y genera figuras
   - Carga CSVs
   - Genera 4 figuras principales en PDF
   - Genera tablas LaTeX
   - Crea reporte de resumen

### Plantillas de Recolección de Datos

Ubicación: `logs/validation_results/`

1. **validation_results_master.csv** - Todos los resultados individuales
2. **validation_matlab_comparison.csv** - Comparación específica con MATLAB
3. **validation_convergence_dz.csv** - Análisis de convergencia
4. **validation_scaling_nsym.csv** - Análisis de escalabilidad

---

## 🚀 WORKFLOW COMPLETO

### ETAPA 1: Preparación (✅ COMPLETADA)

```bash
# Ya están creados:
# - 8 configs JSON
# - Scripts de automatización
# - Plantillas CSV
# - Documentación
```

### ETAPA 2: Ejecución de Simulaciones (📋 TU TAREA)

#### Opción A: Ejecución Manual (recomendada para debugging)

```powershell
# Activar entorno
cd "C:\Users\benja\Desktop\sim fibra\fibra sim\Fibersim_py"
conda activate fibersim

# Verificar setup
python --version
nvidia-smi

# Ejecutar una config individual
python -c "
from src.fibersim.core.simulation import run_simulation
import json
import time

cfg = json.load(open('examples/configs/val_qpsk_40km_baseline.json'))
start = time.time()
results = run_simulation(cfg, backend='cupy')  # o 'numpy'
print(f'Tiempo: {time.time() - start:.2f}s')
print(f'BER: {results[\"ber\"]:.2e}')
"
```

#### Opción B: Ejecución Automática (batch)

**IMPORTANTE**: Primero debes modificar `run_validation_batch.py`:
- Línea ~110: Descomentar `from src.fibersim.core.simulation import run_simulation`
- Línea ~115: Reemplazar simulación mock con llamada real

```powershell
# Ejecutar todas las validaciones
python run_validation_batch.py

# Esto ejecutará ~137 simulaciones y puede tomar:
# - RTX 3060 + CuPy: ~2-4 horas (sin benchmark 250km)
# - Benchmark 250 km: ~15-30 minutos adicional
```

#### ⚠️ IMPORTANTE: Benchmark 250 km

El benchmark de 250 km (dz=1m) es **CRÍTICO** pero **MUY LARGO**:
- MATLAB original: 6.1 horas
- Python + NumPy: ~4-5 horas estimadas
- Python + CuPy (RTX 3060): ~15-30 minutos estimados

**Recomendación**: Ejecutar el benchmark 250 km **en ambos computadores** para comparación.

### ETAPA 3: Recolección de Datos

Durante/después de cada simulación:

1. **Logs JSON**: Se guardan automáticamente en `logs/simlog_*.json`
2. **Datos manuales**: Anota en CSV:
   - Timestamp inicio/fin
   - Plataforma (RTX3060/RTX4060)
   - Backend (numpy/cupy)
   - Tiempo total de ejecución
   - BER, OSNR, EVM (del log)

3. **Monitoreo GPU** (opcional pero útil):
```powershell
# En terminal separado
nvidia-smi -l 1  # Actualiza cada 1s
# Anota utilización GPU y memoria usada
```

### ETAPA 4: Procesamiento de Resultados

```powershell
# Una vez completadas las simulaciones:
python process_validation_results.py

# Esto genera:
# - Latex/MEMORIA/figuras/*.pdf (4 figuras)
# - Latex/MEMORIA/tablas_generadas/*.tex (tablas)
# - validation_summary_report.txt (resumen)
```

### ETAPA 5: Integración en LaTeX

1. Abre `GUIA_INTEGRACION_LATEX.md`
2. Sigue la estructura propuesta para cada sección
3. Copia las tablas de `Latex/MEMORIA/tablas_generadas/`
4. Referencia las figuras de `Latex/MEMORIA/figuras/`
5. Escribe el texto narrativo conectando resultados

```bash
cd Latex/MEMORIA
pdflatex Main.tex
bibtex Main
pdflatex Main.tex
pdflatex Main.tex
```

---

## 📊 MÉTRICAS CLAVE A CAPTURAR

### Desempeño Computacional
- ⏱️ **Tiempo de ejecución** (s)
- 🚀 **Speedup GPU vs CPU** (CuPy/NumPy)
- 📈 **Speedup vs MATLAB** (crítico para justificar migración)
- 💾 **Uso de memoria GPU** (MB)
- 🔢 **Throughput** (Msym/s)

### Validación Física
- 📉 **BER** (Bit Error Rate)
- 📶 **OSNR** (dB)
- 📊 **EVM** (Error Vector Magnitude, %)
- ✅ **Comparación con curvas teóricas** (AWGN)

### Convergencia Numérica
- 🎯 **Error relativo vs dz** (%)
- ⚖️ **Trade-off precisión vs tiempo**
- 📌 **Valor óptimo de dz recomendado**

### Escalabilidad
- 📐 **Complejidad temporal**: Ajuste O(N log N)
- 📏 **Escalabilidad lineal vs distancia**
- 🔁 **Consistencia entre réplicas**

---

## ✅ CHECKLIST DE VALIDACIÓN

### Pre-ejecución
- [ ] Ambiente conda activado
- [ ] GPU disponible (nvidia-smi funciona)
- [ ] Configs JSON revisados y validados
- [ ] Directorio logs/validation_results/ creado
- [ ] Script run_validation_batch.py modificado (quitar mock)

### Durante ejecución
- [ ] Monitorear temperatura GPU (nvidia-smi)
- [ ] Verificar logs se están guardando
- [ ] Anotar tiempos manualmente (backup)
- [ ] Esperar cool-down entre runs largos

### Post-ejecución
- [ ] Verificar que todos los CSVs tienen datos
- [ ] Ejecutar process_validation_results.py
- [ ] Revisar figuras generadas
- [ ] Verificar que tablas LaTeX compilan
- [ ] Calcular promedios y desviaciones estándar

### Integración en memoria
- [ ] Tablas insertadas en 4-Resultados_nuevo.tex
- [ ] Figuras referenciadas correctamente
- [ ] Texto narrativo escrito
- [ ] Referencias cruzadas verificadas
- [ ] LaTeX compila sin errores
- [ ] Revisión ortográfica

---

## 🎓 VALIDACIONES PROPUESTAS

### 1. Validación de Desempeño
**Pregunta**: ¿Qué tan rápido es Python+GPU vs MATLAB?
**Config**: val_benchmark_250km_matlab.json
**Resultado esperado**: Speedup >20x con CuPy

### 2. Validación de Convergencia
**Pregunta**: ¿Qué valor de dz es suficiente?
**Config**: val_qpsk_stepsize_comparison.json
**Resultado esperado**: dz=50-100m con error <5%

### 3. Validación de Escalabilidad
**Pregunta**: ¿Cómo escala el tiempo con Nsym?
**Config**: val_scaling_nsym.json
**Resultado esperado**: Complejidad O(N log N)

### 4. Validación Física: Modulaciones
**Pregunta**: ¿Se reproducen curvas BER vs OSNR teóricas?
**Configs**: val_qpsk_40km, val_16qam_80km, val_64qam_40km
**Resultado esperado**: Penalidad 2-3 dB vs AWGN

### 5. Validación de Sensibilidad
**Pregunta**: ¿Cómo afecta Ptx al trade-off lineal/ruido?
**Config**: val_16qam_80km (variar Ptx)
**Resultado esperado**: Curva en U con óptimo

### 6. Validación de DCF
**Pregunta**: ¿Cuál es la compensación óptima?
**Config**: val_16qam_160km_compensated (variar L_dcf)
**Resultado esperado**: Óptimo en 80-90%

### 7. Validación de Acumulación ASE
**Pregunta**: ¿Se degrada correctamente OSNR con múltiples EDFAs?
**Config**: val_qpsk_120km_3spans (variar nsp)
**Resultado esperado**: Degradación acumulativa consistente

---

## 📈 RESULTADOS ESPERADOS (Estimados)

### Speedup GPU vs CPU
- **RTX 3060**: 18-22x
- **RTX 4060**: 20-25x (10-15% mejor)

### Benchmark MATLAB 250 km
- **MATLAB**: 22,000s (6.1h) [dato real de tu memoria]
- **Python + NumPy**: ~15,000s (4.2h) estimado
- **Python + CuPy (RTX 3060)**: ~900s (15 min) estimado
- **Python + CuPy (RTX 4060)**: ~750s (12.5 min) estimado

### Convergencia dz
- **dz ≤ 50m**: Error <5%
- **dz = 100m**: Error ~5-10% (aceptable para exploración)
- **dz ≥ 200m**: Error >10% (no recomendado)

---

## 🐛 TROUBLESHOOTING

### Error: "CUDA out of memory"
**Solución**: Reducir Nsym o sps en config

### Error: "Module not found: fibersim"
**Solución**: Verificar PYTHONPATH
```powershell
$env:PYTHONPATH="C:\Users\benja\Desktop\sim fibra\fibra sim\Fibersim_py\src"
```

### Error: Simulación muy lenta en GPU
**Solución**: Verificar que CuPy está usando GPU
```python
import cupy as cp
print(cp.cuda.Device(0).compute_capability)  # Debe mostrar (8,6) para RTX 3060
```

### Benchmark 250 km toma demasiado
**Solución**: Ejecutar solo con CuPy, skip NumPy para este caso

---

## 📞 NEXT STEPS

1. **Inmediato**: Modificar `run_validation_batch.py` para usar tu simulador real
2. **Día 1-2**: Ejecutar validaciones Grupo A y B en ambos computadores
3. **Día 3**: Ejecutar Grupo C (convergencia y scaling)
4. **Día 4-5**: Ejecutar benchmark 250 km (¡CRÍTICO!)
5. **Día 6**: Procesar resultados con `process_validation_results.py`
6. **Día 7-8**: Integrar en LaTeX siguiendo GUIA_INTEGRACION_LATEX.md
7. **Día 9**: Revisión final y compilación de memoria

---

## 📚 REFERENCIAS

- **PLAN_VALIDACION.md**: Plan maestro detallado
- **GUIA_INTEGRACION_LATEX.md**: Guía de integración en memoria
- **Capítulo 4 (4-Resultados_nuevo.tex)**: Estructura actual de resultados

---

**¡Éxito con tu trabajo de título!** 🎓🚀

**Preparado por**: GitHub Copilot  
**Fecha**: 26 Nov 2025  
**Versión**: 1.0
