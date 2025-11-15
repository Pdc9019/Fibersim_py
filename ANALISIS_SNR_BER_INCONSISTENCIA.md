# Análisis de Inconsistencia SNR vs BER en FiberSim

## Problema Identificado

El usuario reporta que hay una gran inconsistencia entre los valores de SNR (muy bajos) y BER (muy bajos) en las simulaciones. Esto sugiere que diferentes métricas están usando diferentes definiciones o métodos de cálculo.

## Análisis de Cálculos en el Código

### 1. SNR de Símbolo (`_snr_sym_db` en main.py)

```python
def _snr_sym_db(tx_ref_np, rx_syms_np) -> float:
    \"\"\"SNR a nivel de símbolo tras alinear fase al MMSE.\"\"\"
    import numpy as np
    n = min(len(tx_ref_np), len(rx_syms_np))
    if n == 0:
        return float(\"nan\")
    tx = tx_ref_np[:n]
    rx = rx_syms_np[:n]
    num = np.vdot(tx, rx)  # conj(tx) @ rx
    theta = float(np.angle(num))
    rx_rot = rx * np.exp(-1j * theta)
    err = rx_rot - tx
    ps = float(np.mean(np.abs(tx) ** 2))  # Potencia de señal
    pn = float(np.mean(np.abs(err) ** 2))  # Potencia de ruido
    if pn <= 0:
        return float(\"inf\")
    return 10.0 * math.log10(ps / pn)  # SNR = 10*log10(Ps/Pn)
```

**Características:**
- Calcula SNR comparando símbolos TX vs RX después de TODA la cadena
- Incluye ALL IMPAIRMENTS: ruido ASE, distorsiones no lineales, timing jitter, etc.
- Es una medida \"real\" post-DSP

### 2. OSNR Analítico (usado para BER teórico)

En `gui/app.py` líneas 534-535:
```python
OSNR_lin = 10.0**(osnr/10.0); SNR_lin = OSNR_lin * (Bo_Hz / max(Rb, 1.0))
pt[\"BER\"] = 0.5 * erfc(sqrt(max(SNR_lin,1e-12))/sqrt(2.0))
```

**Características:**
- OSNR calculado solo considerando ruido ASE de amplificadores
- No incluye distorsiones no lineales (fibra), timing jitter, imperfecciones DSP
- Conversión simplificada: `SNR_eléctrico = OSNR_óptico × (Bo/Rb)`
- BER teórico para AWGN ideal (solo ruido gaussiano)

### 3. EVM como Métrica Intermedia

En `modem.py`:
```python
def evm_db(y: np.ndarray, ref: np.ndarray) -> Tuple[float, float]:
    # ... alineación de fase ...
    err = y_rot - ref
    Ps = float(np.mean(np.abs(ref) ** 2))
    Pe = float(np.mean(np.abs(err) ** 2))
    evm_rms = np.sqrt(Pe / Ps)
    evm_db_val = 20.0 * np.log10(max(evm_rms, 1e-15))
```

## Fuentes de la Inconsistencia

### 1. **Diferentes Definiciones de \"Ruido\"**

- **SNR símbolo**: Incluye TODO el ruido/distorsión (ASE + nonlinear + timing + DSP)
- **OSNR → BER**: Solo considera ruido ASE, asume canal AWGN ideal

### 2. **Efectos No Lineales No Considerados en BER**

El BER teórico usa:
```python
BER = 0.5 * erfc(sqrt(SNR)/sqrt(2.0))  # Fórmula AWGN
```

Pero en fibra óptica real:
- Dispersión cromática → distorsión intersímbolo
- Efectos Kerr (γ ≠ 0) → distorsión de amplitud/fase
- PMD, PDL → distorsión de polarización
- Timing jitter del receptor

### 3. **Mapeo OSNR → SNR Eléctrico Simplificado**

```python
SNR_lin = OSNR_lin * (Bo_Hz / max(Rb, 1.0))
```

Esta conversión asume:
- Detector ideal (sin ruido térmico/shot)
- Filtrado óptico perfecto
- No hay penalty por imperfecciones del receptor

### 4. **Ancho de Banda de Referencia**

- OSNR se mide típicamente en 0.1 nm (≈12.5 GHz @ 1550 nm)
- SNR eléctrico debería referirse al ancho de banda del símbolo
- La conversión puede estar subestimando el penalty

## Ejemplo Numérico del Problema

**Escenario típico:**
- OSNR = 20 dB (bueno)
- Bo = 12.5 GHz, Rb = 32 GBaud
- SNR_teórico = 20 + 10*log10(12.5/32) = 20 - 4.1 = 15.9 dB
- BER_teórico = 0.5 * erfc(sqrt(10^1.59)/sqrt(2)) ≈ 1e-6 (muy bajo)

**Pero en la simulación real:**
- SNR_símbolo = 8 dB (porque incluye nonlinear effects, timing, etc.)
- Esta diferencia de ~8 dB indica penalty significativo

## Recomendaciones para Corregir

### 1. **Usar SNR Post-DSP Consistente**

Reemplazar el cálculo de BER teórico con el SNR real medido:

```python
# En lugar de usar OSNR analítico
SNR_lin = OSNR_lin * (Bo_Hz / max(Rb, 1.0))

# Usar SNR medido post-DSP
SNR_measured_dB = res.get(\"SNR_sym_dB\", None)
if SNR_measured_dB is not None:
    SNR_lin = 10.0**(SNR_measured_dB/10.0)
```

### 2. **Agregar Penalty Models**

Para el BER analítico, añadir penalties típicos:

```python
# Penalty por efectos no lineales (función de potencia y distancia)
nonlinear_penalty_dB = estimate_nonlinear_penalty(power_dBm, fiber_length_km, gamma)

# Penalty por dispersión
dispersion_penalty_dB = estimate_dispersion_penalty(beta2, length, bit_rate)

# SNR efectivo
SNR_effective_dB = SNR_from_OSNR_dB - nonlinear_penalty_dB - dispersion_penalty_dB
```

### 3. **Mostrar Ambas Métricas Claramente**

En la GUI, distinguir:
- \"OSNR (solo ASE)\": valor analítico
- \"SNR símbolo (post-DSP)\": valor medido real
- \"Penalty total\": diferencia entre ambos

### 4. **Validar Conversión OSNR→SNR**

La fórmula actual puede estar incorrecta para sistemas WDM o con filtrado no ideal.

## Archivos a Modificar

1. `gui/app.py` - Líneas 532-535: cálculo de BER
2. `main.py` - Reportar penalty además de SNR absoluto
3. `analysis/profile.py` - Usar SNR medido en lugar de OSNR teórico
4. Agregar función de penalty models

## Conclusión

La inconsistencia surge porque:
- **SNR reportado**: medida real incluyendo todas las imperfecciones
- **BER calculado**: teórico basado solo en ruido ASE

Para sistemas de fibra óptica reales, el penalty entre OSNR y SNR efectivo puede ser 5-15 dB dependiendo de:
- Efectos no lineales (alta potencia)
- Dispersión acumulada
- Calidad del DSP/timing recovery
- Imperfecciones del transceiver

La solución es usar consistentemente el SNR medido post-DSP para todos los cálculos de BER, o agregar modelos de penalty realistas al cálculo teórico.
