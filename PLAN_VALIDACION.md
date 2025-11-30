# PLAN DE VALIDACIÓN Y EVALUACIÓN DE DESEMPEÑO
## Fibersim_py - Trabajo de Título

---

## 1. OBJETIVOS DE LA VALIDACIÓN

### 1.1 Validación Funcional
- **Exactitud del modelo SSFM**: Verificar convergencia numérica vs step size (dz)
- **Cálculo correcto de OSNR**: Comparar valores teóricos vs simulados
- **Modelo de ruido ASE**: Validar acumulación de ruido en cascadas de EDFAs
- **BER vs OSNR**: Verificar curvas teóricas para QPSK, 16-QAM, 64-QAM

### 1.2 Validación de Desempeño
- **Speedup GPU vs CPU**: Cuantificar aceleración CuPy vs NumPy
- **Escalabilidad**: Analizar complejidad temporal vs N_sym, distancia, dz
- **Comparación MATLAB**: Benchmark directo con prototipo original
- **Throughput**: Símbolos procesados por segundo

### 1.3 Análisis de Sensibilidad
- **Potencia de transmisión (Ptx)**: Impacto en no-linealidad vs OSNR
- **Step size (dz)**: Trade-off precisión vs tiempo de cómputo
- **Figura de ruido (nsp)**: Degradación acumulativa en enlaces largos
- **Compensación de dispersión**: Efecto de DCF en BER/EVM

---

## 2. CONFIGURACIONES DE PRUEBA

### Grupo A: Validación por Modulación
1. **val_qpsk_40km_baseline.json** - QPSK baseline
   - Variar: Ptx = [0.1, 1, 5, 10] mW
   - Métricas: BER, OSNR, tiempo

2. **val_16qam_80km_baseline.json** - 16-QAM baseline
   - Variar: Ptx = [1, 2, 5, 10] mW
   - Métricas: EVM, BER, OSNR, tiempo

3. **val_64qam_40km_highsnr.json** - 64-QAM alto orden
   - Variar: nsp = [1.2, 1.5, 2.0]
   - Métricas: EVM, BER, umbral OSNR

### Grupo B: Validación de Escalabilidad
4. **val_qpsk_120km_3spans.json** - Acumulación de ruido
   - Variar: nsp = [1.5, 2.0, 2.5, 3.0]
   - Métricas: OSNR acumulado, BER

5. **val_16qam_160km_compensated.json** - DCF
   - Variar: L_dcf = [6, 8, 10] km (60%, 80%, 100% comp.)
   - Métricas: Dispersión residual, BER, EVM

### Grupo C: Convergencia Numérica
6. **val_qpsk_stepsize_comparison.json** - Convergencia SSFM
   - Variar: dz = [10, 25, 50, 100, 200, 500, 1000] m
   - Métricas: BER vs dz, tiempo vs dz, error relativo

7. **val_scaling_nsym.json** - Complejidad computacional
   - Variar: Nsym = [1024, 2048, 4096, 8192, 16384, 32768]
   - Métricas: tiempo vs Nsym, speedup, memoria

### Grupo D: Benchmark MATLAB
8. **val_benchmark_250km_matlab.json** - Comparación directa
   - Config fija (dz=1m)
   - Comparar: MATLAB (6h) vs NumPy vs CuPy (RTX 3060 y RTX 4060)

---

## 3. MATRIZ DE PRUEBAS

| Config | Plataforma | Backend | Variaciones | Réplicas | Total Runs |
|--------|------------|---------|-------------|----------|------------|
| val_qpsk_40km_baseline | Ambas | NumPy + CuPy | 4 Ptx | 3 | 24 |
| val_16qam_80km_baseline | Ambas | NumPy + CuPy | 4 Ptx | 3 | 24 |
| val_64qam_40km_highsnr | Ambas | NumPy + CuPy | 3 nsp | 3 | 18 |
| val_qpsk_120km_3spans | Ambas | NumPy + CuPy | 4 nsp | 3 | 24 |
| val_16qam_160km_compensated | Ambas | NumPy + CuPy | 3 L_dcf | 2 | 12 |
| val_qpsk_stepsize_comparison | RTX 3060 | CuPy | 7 dz | 1 | 7 |
| val_scaling_nsym | Ambas | NumPy + CuPy | 6 Nsym | 2 | 24 |
| val_benchmark_250km_matlab | Ambas | NumPy + CuPy | 1 | 1 | 4 |
| **TOTAL** | | | | | **137 simulaciones** |

**Plataformas:**
- **RTX 3060**: Intel i7-12700H, RTX 3060 6GB, 16GB RAM
- **RTX 4060**: Intel Ultra 7 115H, RTX 4060 8GB, 16GB RAM

---

## 4. MÉTRICAS A RECOLECTAR

### 4.1 Desde Logs JSON (simlog_*.json)
```json
{
  "timestamp": "...",
  "config_file": "...",
  "platform": "RTX3060 | RTX4060",
  "backend": "numpy | cupy",
  "execution_time_s": ...,
  "Nsym": ...,
  "sps": ...,
  "total_samples": ...,
  "chain_length_km": ...,
  "num_steps_ssfm": ...,
  "results": {
    "ber": ...,
    "osnr_db": ...,
    "evm_percent": ...,
    "rx_power_dbm": ...,
    "constellation_data": ...
  }
}
```

### 4.2 Métricas Derivadas
- **Speedup**: `T_numpy / T_cupy`
- **Speedup vs MATLAB**: `T_matlab / T_python`
- **Throughput**: `Nsym / execution_time_s` [Msym/s]
- **Eficiencia GPU**: Comparar RTX 3060 vs RTX 4060
- **Convergencia**: Error relativo vs dz de referencia

### 4.3 Hardware Utilization (Manual)
- GPU utilization % (nvidia-smi durante ejecución)
- Memoria GPU usada (MB)
- CPU usage %
- RAM usage (GB)

---

## 5. PROCEDIMIENTO DE EJECUCIÓN

### 5.1 Preparación
```powershell
# Activar entorno
cd "C:\Users\benja\Desktop\sim fibra\fibra sim\Fibersim_py"
conda activate fibersim

# Verificar versiones
python --version
python -c "import numpy; print(numpy.__version__)"
python -c "import cupy; print(cupy.__version__)"
nvidia-smi
```

### 5.2 Ejecución Individual
```powershell
# Ejemplo: QPSK baseline con Ptx=1mW, NumPy, RTX 3060
# 1. Editar val_qpsk_40km_baseline.json: "Ptx": 0.001
# 2. Ejecutar
python -c "
from src.fibersim.core.simulation import run_simulation
import json
import time

cfg = json.load(open('examples/configs/val_qpsk_40km_baseline.json'))
start = time.time()
results = run_simulation(cfg, backend='numpy')
elapsed = time.time() - start
print(f'Time: {elapsed:.2f} s')
print(f'BER: {results[\"ber\"]:.2e}')
print(f'OSNR: {results[\"osnr_db\"]:.2f} dB')
"
```

### 5.3 Automatización (Script de Batch)
Crear script Python para automatizar las 137 simulaciones con:
- Carga de config base
- Modificación programática de parámetros
- Iteración sobre variaciones
- Logging automático con metadatos (plataforma, backend, timestamp)
- Almacenamiento en `logs/validation_results/`

---

## 6. FORMATO DE RECOLECCIÓN DE DATOS

### 6.1 Tabla Master - Resultados Completos
**Archivo**: `validation_results_master.csv`

| run_id | timestamp | platform | backend | config | param_varied | param_value | Nsym | sps | L_km | dz_m | num_steps | exec_time_s | throughput_Msps | ber | osnr_db | evm_pct | speedup | notes |
|--------|-----------|----------|---------|--------|--------------|-------------|------|-----|------|------|-----------|-------------|-----------------|-----|---------|---------|---------|-------|
| 001 | 2025-11-26_14:30:15 | RTX3060 | numpy | val_qpsk_40km | Ptx | 0.001 | 8192 | 16 | 40 | 100 | 400 | 45.3 | 180.8 | 1.2e-3 | 15.3 | - | 1.0x | baseline |
| 002 | 2025-11-26_14:32:45 | RTX3060 | cupy | val_qpsk_40km | Ptx | 0.001 | 8192 | 16 | 40 | 100 | 400 | 2.1 | 3901.0 | 1.2e-3 | 15.3 | - | 21.6x | GPU |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

### 6.2 Tabla Comparativa - Benchmark MATLAB
**Archivo**: `validation_matlab_comparison.csv`

| platform | backend | exec_time_s | exec_time_h | speedup_vs_matlab | ber | osnr_db | notes |
|----------|---------|-------------|-------------|-------------------|-----|---------|-------|
| MATLAB | - | 22000 | 6.1 | 1.0x | - | - | Referencia original |
| RTX3060 | numpy | 18500 | 5.1 | 1.19x | - | - | Python CPU |
| RTX3060 | cupy | 950 | 0.26 | 23.2x | - | - | Python GPU |
| RTX4060 | numpy | 16800 | 4.7 | 1.31x | - | - | CPU más rápido |
| RTX4060 | cupy | 720 | 0.20 | 30.6x | - | - | GPU más rápida |

### 6.3 Tabla Análisis de Convergencia - Step Size
**Archivo**: `validation_convergence_dz.csv`

| dz_m | num_steps | exec_time_cupy_s | ber | osnr_db | rel_error_ber_pct | rel_error_osnr_db | converged |
|------|-----------|------------------|-----|---------|-------------------|-------------------|-----------|
| 1 | 80000 | 2850 | 1.23e-3 | 15.34 | 0.0 | 0.0 | reference |
| 10 | 8000 | 285 | 1.24e-3 | 15.32 | 0.8 | -0.13 | yes |
| 25 | 3200 | 115 | 1.26e-3 | 15.28 | 2.4 | -0.39 | yes |
| 50 | 1600 | 58 | 1.31e-3 | 15.19 | 6.5 | -0.98 | marginal |
| 100 | 800 | 30 | 1.45e-3 | 14.89 | 17.9 | -2.93 | no |
| ... | ... | ... | ... | ... | ... | ... | ... |

---

## 7. ANÁLISIS ESTADÍSTICO

### 7.1 Métricas de Centralidad
- Media, mediana, desviación estándar para tiempos de ejecución (réplicas)
- Intervalos de confianza 95% para speedup

### 7.2 Regresiones
- **Escalabilidad Temporal**: Ajustar `T = a*N*log(N) + b` para validar complejidad FFT
- **Convergencia dz**: Ajustar decaimiento exponencial del error

### 7.3 Validación Teórica
- **BER vs OSNR**: Comparar con curvas teóricas de Gray-coded M-QAM
- **OSNR acumulado**: Validar fórmula de Personick para cascadas de EDFAs
- **Dispersión acumulada**: Verificar compensación exacta con DCF

---

## 8. ENTREGABLES

### 8.1 Datasets
1. `validation_results_master.csv` - Todos los runs
2. `validation_matlab_comparison.csv` - Benchmark específico
3. `validation_convergence_dz.csv` - Análisis de convergencia
4. `validation_scaling_nsym.csv` - Análisis de escalabilidad
5. Logs JSON individuales en `logs/validation_results/`

### 8.2 Figuras para la Memoria
1. **Speedup GPU vs CPU** (bar chart, ambas plataformas)
2. **Benchmark MATLAB comparison** (bar chart horizontal)
3. **BER vs OSNR** para QPSK, 16-QAM, 64-QAM con curvas teóricas
4. **Convergencia dz**: BER/tiempo vs step size (dual-axis)
5. **Escalabilidad**: log-log plot tiempo vs Nsym con ajuste O(N log N)
6. **Efecto de Ptx**: BER vs potencia para mostrar trade-off lineal/ruido
7. **Acumulación ASE**: OSNR vs número de spans
8. **Compensación DCF**: EVM vs % compensación

### 8.3 Tablas para la Memoria
1. Resumen de configuraciones de prueba
2. Especificaciones de hardware (ya existe en LaTeX)
3. Comparación de tiempos MATLAB vs Python
4. Speedup promedio por configuración
5. Umbrales de BER para cada modulación

---

## 9. CALENDARIO DE EJECUCIÓN

### Semana 1
- **Día 1-2**: Ejecutar Grupo A (modulaciones) en ambas plataformas
- **Día 3**: Ejecutar Grupo B (escalabilidad espacial)

### Semana 2
- **Día 1**: Ejecutar Grupo C (convergencia numérica)
- **Día 2-3**: Ejecutar Grupo D (benchmark MATLAB 250 km - largo!)

### Semana 3
- **Día 1**: Procesamiento de datos, generación de CSVs consolidados
- **Día 2-3**: Generación de figuras y análisis estadístico
- **Día 4**: Integración en LaTeX (Capítulo 4 - Resultados)

---

## 10. CHECKLIST PRE-EJECUCIÓN

- [ ] Ambiente conda `fibersim` activo
- [ ] Versiones verificadas (Python 3.12.7, NumPy 1.26.4, CuPy 12.9)
- [ ] GPU drivers actualizados (CUDA 12.x)
- [ ] Directorio `logs/validation_results/` creado
- [ ] Configs JSON validados (sintaxis JSON correcta)
- [ ] Script de batch testing preparado
- [ ] Herramientas de monitoreo: `nvidia-smi`, Task Manager
- [ ] Backup de datos previos

---

## 11. NOTAS IMPORTANTES

1. **Réplicas**: Ejecutar 2-3 réplicas para promediar variabilidad térmica de GPU
2. **Cool-down**: Esperar 30s entre runs consecutivos en GPU
3. **Logging**: Asegurar que cada log incluya metadatos completos (platform, backend, config)
4. **Interrupciones**: Si se interrumpe benchmark 250km, documentar tiempo parcial
5. **Memoria GPU**: Monitorear para Nsym grandes; si OOM, reducir batch size
6. **Comparación justa**: Mismo config entre NumPy/CuPy/platforms

---

**Preparado por**: GitHub Copilot  
**Fecha**: 26 Nov 2025  
**Versión**: 1.0
