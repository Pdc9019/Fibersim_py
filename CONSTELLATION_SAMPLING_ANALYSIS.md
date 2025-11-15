# Análisis: Muestreo de Constelaciones en FiberSim

## Resumen Ejecutivo

El simulador **ya está muestreando correctamente** (1 punto por símbolo), pero puede beneficiarse de **limitar el número de puntos mostrados** para mejor visualización.

## Estado Actual del Código

### ✓ YA IMPLEMENTADO en chain.py (línea 66, 78, 88)
```python
sym = B[delay::sps]  # Solo 1 muestra cada 'sps' muestras
```

Esto ya está **correcto** - toma 1 punto por símbolo en el instante óptimo de muestreo.

### ✓ MEJORADO en plot.py
```python
def save_constellations_grid(..., max_symbols: int = 2000):
    # Limita símbolos mostrados sin perder información visual
    if len(s) > max_symbols:
        step = len(s) // max_symbols
        s = s[::step]
```

## Comparación de Métodos

### 1. Sobremuestreado (16 puntos/símbolo) ❌
- **Puntos mostrados**: N × 16 (ej: 8192 × 16 = 131,072 puntos)
- **Apariencia**: LÍNEAS horizontales `▬▬▬`
- **Problema**: Difícil ver la distribución real del ruido
- **Causa**: Muestra la trayectoria temporal del pulso RRC

### 2. 1 Punto por Símbolo ✓
- **Puntos mostrados**: N (ej: 8192 puntos)
- **Apariencia**: NUBES bien definidas `☁`
- **Ventaja**: Muestra claramente la distribución gaussiana del ruido
- **Estado**: YA IMPLEMENTADO en el código actual

### 3. Diezmado Adicional (Recomendado) ✓✓
- **Puntos mostrados**: ~2000 puntos
- **Apariencia**: NUBES claras y nítidas
- **Ventaja**: Visualización óptima + renderizado rápido
- **Estado**: RECIÉN IMPLEMENTADO en plot.py

## Resultados de la Prueba

```
Configuración: QPSK, SNR = 15 dB

┌────────────────────┬─────────────┬──────────────┬─────────────────┐
│ Método             │ Puntos      │ Apariencia   │ Recomendación   │
├────────────────────┼─────────────┼──────────────┼─────────────────┤
│ Sobremuestreado    │ 16,000      │ Líneas ▬▬▬   │ ❌ No usar      │
│ 1 por símbolo      │ 1,000       │ Nubes ☁☁     │ ✓ Funcional     │
│ Diezmado (1/5)     │ 200         │ Nubes claras │ ✓✓ Óptimo       │
└────────────────────┴─────────────┴──────────────┴─────────────────┘
```

## Explicación Física

### Por qué se ven LÍNEAS con sobremuestreo:

1. **Filtro RRC**: Cada símbolo no es un punto instantáneo, sino un pulso que se extiende en el tiempo
2. **Oversampling**: 16 muestras capturan la evolución temporal del pulso
3. **En I/Q**: La trayectoria temporal del pulso aparece como una línea

```
Dominio Temporal:          Plano I/Q:
                          
    ╱──╲                     │
   ╱    ╲                    │   ▬▬▬▬▬  (línea)
  ╱      ╲                   │
 ╱        ╲                  └──────────
 Pulso RRC                   Trayectoria del pulso
 16 muestras                 en el espacio I/Q
```

### Por qué se ven NUBES con 1 punto/símbolo:

1. **Muestreo óptimo**: Solo en el pico del pulso (máxima apertura del ojo)
2. **Ruido gaussiano**: Cada símbolo tiene un error aleatorio independiente
3. **Distribución**: Los puntos forman nubes gaussianas alrededor de los símbolos ideales

```
Símbolo ideal: •

Con ruido gaussiano:
    
    ☁ ☁ ☁
   ☁ ☁• ☁ ☁
    ☁ ☁ ☁
    
Distribución normal 2D
```

## Métricas y Visualización

### SNR y tamaño de las nubes

```
SNR alto (20 dB):      SNR medio (15 dB):     SNR bajo (10 dB):
   ☁                      ☁☁                     ☁☁☁
  • (compacto)           •• (medio)            ☁☁•☁☁ (disperso)
   ☁                      ☁☁                     ☁☁☁
```

### BER y solapamiento

```
BER bajo (< 10⁻⁴):     BER medio (10⁻³):      BER alto (10⁻²):
  ☁   ☁                  ☁ ☁                    ☁☁☁☁
  •   •                  • •                    ☁☁☁☁
  ☁   ☁                  ☁ ☁                    (solapadas)
(separadas)            (tocándose)            → Muchos errores
```

## Recomendaciones Finales

### Para el simulador FiberSim:

1. ✅ **Mantener** el muestreo actual (1 punto/símbolo en chain.py)
2. ✅ **Usar** la nueva función mejorada de plot.py con `max_symbols=2000`
3. ✅ **Mostrar** número de puntos en el título: `"50 km (N=2000)"`

### Configuración óptima por escenario:

```python
# Análisis rápido / GUI interactiva
max_symbols = 1000  # Rápido, suficiente para visualización

# Análisis detallado / publicaciones
max_symbols = 5000  # Mayor resolución estadística

# Debugging / verificación
max_symbols = 200   # Ultra rápido, solo para verificar forma
```

## Conclusión

El código **ya está correcto** desde el punto de vista técnico. La mejora implementada (limitación a 2000 puntos) solo optimiza la **visualización** sin perder información relevante sobre:
- Distribución del ruido
- SNR del sistema
- Calidad de la señal
- Efectos de degradación

La constelación con 2000 puntos es:
- Estadísticamente representativa
- Visualmente clara
- Computacionalmente eficiente
- Perfecta para análisis de enlaces ópticos

---
**Fecha**: Noviembre 2025
**Implementado en**: plot.py (save_constellations_grid)
**Estado**: ✅ Listo para producción
