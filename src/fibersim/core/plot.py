from __future__ import annotations
from typing import Sequence
import matplotlib.pyplot as plt
from pathlib import Path
from .array_api import xp
from .array_api import asnumpy as _asnp

def save_constellations_grid(consSym: Sequence, consZ_m: Sequence[float], outpath: Path, 
                            max_symbols: int = 500):
    """
    Guarda un grid de constelaciones con opción de limitar el número de símbolos mostrados.
    
    Args:
        consSym: Lista de arrays de símbolos para cada punto de captura
        consZ_m: Distancias en metros para cada punto
        outpath: Ruta de salida para el PNG
        max_symbols: Máximo número de símbolos a mostrar por constelación (default: 500)
    """
    n = len(consSym)
    if n == 0:
        return
    rows = int((n - 1) ** 0.5) + 1
    cols = rows
    fig = plt.figure(figsize=(cols * 3.2, rows * 3.2))
    for i, (sym, z) in enumerate(zip(consSym, consZ_m), start=1):
        ax = fig.add_subplot(rows, cols, i)
        s = _asnp(sym)
        
        # Limitar número de símbolos mostrados para mejor visualización
        if len(s) > max_symbols:
            # Diezmar tomando símbolos espaciados uniformemente
            step = len(s) // max_symbols
            s = s[::step]
        
        # Usar marcadores circulares más visibles
        ax.plot(s.real, s.imag, "o", markersize=3, alpha=0.6)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{z/1e3:.0f} km (N={len(s)})")
    fig.suptitle("Evolución de constelación (símbolos muestreados)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close(fig)

def save_eyediagram(Arx, sps: int, delay: int, outpath: Path, spans: int = 2_000):
    y = _asnp(Arx[delay:])
    sps = int(sps)
    nsym = min(spans, len(y) // sps - 2)
    fig = plt.figure(figsize=(7, 4))
    for i in range(nsym):
        seg = y[i * sps : (i + 2) * sps]
        plt.plot(seg.real)  # solo componente real (BPSK)
    plt.title("Eye diagram")
    plt.xlabel("Muestras")
    plt.ylabel("Amplitud")
    plt.grid(True)
    plt.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close(fig)

def save_power_evolution(powZ_m, powW_W, outpath: Path, unit: str = "dBm"):
    import numpy as np
    z = np.asarray(powZ_m) / 1e3
    p = np.asarray(powW_W)
    if unit.lower() == "dbm":
        p_plot = 10 * np.log10(np.maximum(p, 1e-15) / 1e-3)
        ylab = "Potencia (dBm)"
    elif unit.lower() == "mw":
        p_plot = p * 1e3
        ylab = "Potencia (mW)"
    else:
        p_plot = p
        ylab = "Potencia (W)"
    fig = plt.figure(figsize=(7, 4))
    plt.plot(z, p_plot, "-o", markersize=3)
    plt.xlabel("Distancia (km)")
    plt.ylabel(ylab)
    plt.grid(True)
    plt.title("Evolución de potencia óptica media")
    plt.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close(fig)

def save_constellations_3d(consSym, consZ_m, outpath: Path,
                           every: int = 1, pts_per_slice: int = 2000, marker_size: float = 1.0):
    """Constelaciones 3D: X=I, Y=Q, Z=distancia (km).
    - every: usa 1 de cada 'every' snapshots (para aligerar).
    - pts_per_slice: máximo de puntos por snapshot.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    import numpy as np
    import matplotlib.pyplot as plt

    if not consSym or len(consSym) == 0:
        return

    xs, ys, zs = [], [], []
    for idx, (sym, z_m) in enumerate(zip(consSym, consZ_m)):
        if (idx % every) != 0:
            continue
        s = _asnp(sym).astype(np.complex128, copy=False)
        if s.size == 0:
            continue
        # submuestreo aleatorio si hay muchos puntos
        if s.size > pts_per_slice:
            sel = np.random.choice(s.size, size=pts_per_slice, replace=False)
            s = s[sel]
        xs.append(s.real)
        ys.append(s.imag)
        zs.append(np.full(s.shape, z_m / 1e3, dtype=np.float64))  # km

    if not xs:
        return

    X = np.concatenate(xs)
    Y = np.concatenate(ys)
    Z = np.concatenate(zs)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X, Y, Z, s=marker_size)
    ax.set_xlabel("In-Phase")
    ax.set_ylabel("Quadrature")
    ax.set_zlabel("Distancia (km)")
    ax.set_title("Evolución 3D de constelaciones a lo largo del enlace")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close(fig)

def save_constellations_3d_html(consSym, consZ_m, outpath: Path,
                                every: int = 1, pts_per_slice: int = 2000, marker_size: float = 2.0,
                                theme: str = "plotly_dark",
                                add_centroids: bool = False,
                                add_slider: bool = True,
                                trace_symbols: bool = True,
                                num_traces: int = 50,
                                show_slice_planes: bool = True,
                                group_by_quadrant: bool = True):
    """
    Visualización 3D interactiva de constelaciones.
    
    Args:
        trace_symbols: Si True, traza trayectorias de símbolos individuales (cuerdas).
                      Si False, usa el modo nube tradicional.
        num_traces: Número de símbolos a seguir (solo si trace_symbols=True).
        show_slice_planes: Si True, muestra planos semitransparentes en z-slices clave.
        group_by_quadrant: Si True, agrupa símbolos por su cuadrante inicial (mismo color por cuadrante).
    """
    import numpy as np
    import plotly.graph_objects as go

    if not consSym or len(consSym) == 0:
        return

    # --- preparar slices ---
    slices = []
    for idx, (sym, z_m) in enumerate(zip(consSym, consZ_m)):
        if (idx % every) != 0:
            continue
        s = _asnp(sym).astype(np.complex128, copy=False)
        if s.size == 0:
            continue
        
        if trace_symbols:
            # Modo trayectorias: NO muestrear aleatoriamente
            # Usar los primeros N símbolos consistentemente
            if s.size > num_traces:
                s = s[:num_traces]
        else:
            # Modo nube: muestreo aleatorio tradicional
            if s.size > pts_per_slice:
                sel = np.random.choice(s.size, size=pts_per_slice, replace=False)
                s = s[sel]
        
        slices.append((s, float(z_m) / 1e3))  # km

    if not slices:
        return

    def lim_rob(a):
        # Usar percentil 99.9 en lugar de 99.5 para evitar cortar trayectorias extremas
        m = float(np.percentile(np.abs(a), 99.9))
        return (-m * 1.1, m * 1.1)  # Añadir 10% de margen extra
    
    def assign_region_adaptive(symbol, modulation_type='QPSK'):
        """
        Asigna un símbolo a una región según la modulación detectada.
        Detecta automáticamente BPSK (2 regiones), QPSK (4 regiones) o 16-QAM (16 regiones)
        
        Args:
            symbol: Símbolo complejo
            modulation_type: 'BPSK', 'QPSK', o '16-QAM' (detectado automáticamente si no se especifica)
        """
        I, Q = symbol.real, symbol.imag
        
        magnitude = np.sqrt(I**2 + Q**2)
        if magnitude < 1e-10:  # Símbolo casi nulo (evitar división por cero)
            return 0
        
        # BPSK: Solo usar cuadrantes derecho/izquierdo (eje real)
        if modulation_type == 'BPSK':
            return 0 if I >= 0 else 1
        
        # Para QPSK: Usar cuadrantes simples (4 regiones)
        if I >= 0 and Q >= 0:
            base = 0  # Q1
        elif I < 0 and Q >= 0:
            base = 1  # Q2
        elif I < 0 and Q < 0:
            base = 2  # Q3
        else:
            base = 3  # Q4
        
        # Para 16-QAM, subdividir cada cuadrante en 4 sub-regiones
        if modulation_type == '16-QAM':
            # Usar magnitudes absolutas de I y Q para detectar nivel interno vs externo
            abs_I, abs_Q = abs(I), abs(Q)
            # Umbral para separar niveles (ajustar según normalización)
            # En 16-QAM normalizado: niveles en ±1, ±3 → umbral en 2
            threshold = 1.5
            inner_I = 1 if abs_I < threshold else 0  # 1 = nivel interno (±1), 0 = nivel externo (±3)
            inner_Q = 1 if abs_Q < threshold else 0
            sub_region = inner_I * 2 + inner_Q  # 0, 1, 2, 3
            return base * 4 + sub_region  # 0-15 (16 regiones distintas)
        
        # QPSK: Solo retornar el cuadrante base
        return base
    
    # Paleta de colores expandida para 16-QAM (16 colores distintos)
    region_colors_16qam = {
        # Cuadrante Q1 (+I, +Q) - Tonos rojos/naranjas
        0: 'rgb(255, 100, 100)',   # Externo-Externo
        1: 'rgb(255, 150, 100)',   # Externo-Interno
        2: 'rgb(255, 100, 150)',   # Interno-Externo
        3: 'rgb(255, 180, 150)',   # Interno-Interno
        
        # Cuadrante Q2 (-I, +Q) - Tonos verdes
        4: 'rgb(100, 255, 100)',   # Externo-Externo
        5: 'rgb(150, 255, 100)',   # Externo-Interno
        6: 'rgb(100, 255, 150)',   # Interno-Externo
        7: 'rgb(150, 255, 180)',   # Interno-Interno
        
        # Cuadrante Q3 (-I, -Q) - Tonos azules
        8: 'rgb(100, 100, 255)',   # Externo-Externo
        9: 'rgb(100, 150, 255)',   # Externo-Interno
        10: 'rgb(150, 100, 255)',  # Interno-Externo
        11: 'rgb(180, 150, 255)',  # Interno-Interno
        
        # Cuadrante Q4 (+I, -Q) - Tonos amarillos/naranjas
        12: 'rgb(255, 255, 100)',  # Externo-Externo
        13: 'rgb(255, 255, 150)',  # Externo-Interno
        14: 'rgb(255, 200, 100)',  # Interno-Externo
        15: 'rgb(255, 220, 150)',  # Interno-Interno
    }
    
    region_colors_qpsk = {
        0: 'rgb(255, 100, 100)',  # Rojo para Q1
        1: 'rgb(100, 255, 100)',  # Verde para Q2
        2: 'rgb(100, 100, 255)',  # Azul para Q3
        3: 'rgb(255, 255, 100)',  # Amarillo para Q4
    }
    
    region_colors_bpsk = {
        0: 'rgb(255, 100, 100)',  # Rojo para +1
        1: 'rgb(100, 100, 255)',  # Azul para -1
    }
    
    region_names_16qam = {
        0: 'Q1-EE (+3,+3)', 1: 'Q1-EI (+3,+1)', 2: 'Q1-IE (+1,+3)', 3: 'Q1-II (+1,+1)',
        4: 'Q2-EE (-3,+3)', 5: 'Q2-EI (-3,+1)', 6: 'Q2-IE (-1,+3)', 7: 'Q2-II (-1,+1)',
        8: 'Q3-EE (-3,-3)', 9: 'Q3-EI (-3,-1)', 10: 'Q3-IE (-1,-3)', 11: 'Q3-II (-1,-1)',
        12: 'Q4-EE (+3,-3)', 13: 'Q4-EI (+3,-1)', 14: 'Q4-IE (+1,-3)', 15: 'Q4-II (+1,-1)',
    }
    
    region_names_qpsk = {
        0: 'Q1 (+I,+Q)',
        1: 'Q2 (-I,+Q)',
        2: 'Q3 (-I,-Q)',
        3: 'Q4 (+I,-Q)',
    }
    
    region_names_bpsk = {
        0: 'Fase +1',
        1: 'Fase -1',
    }
    
    # Detectar modulación mirando la primera constelación
    initial_sample = slices[0][0][:min(100, len(slices[0][0]))]  # Usar más muestras para mejor detección
    
    # Calcular magnitud promedio para normalizar detección
    magnitudes = np.abs(initial_sample)
    avg_magnitude = np.mean(magnitudes[magnitudes > 1e-10])  # Evitar ceros
    
    # Componentes Q relativas (normalizadas por magnitud)
    Q_relative = np.abs(initial_sample.imag) / (avg_magnitude + 1e-10)
    
    # BPSK: todos los símbolos tienen Q << magnitud (están en eje real)
    is_bpsk = np.mean(Q_relative) < 0.20  # Si Q promedio < 20% de magnitud → BPSK
    
    # 16-QAM: detectar si hay más de 2 niveles distintos en I o Q
    # Normalizar y redondear para contar niveles
    I_normalized = initial_sample.real / (avg_magnitude + 1e-10)
    Q_normalized = initial_sample.imag / (avg_magnitude + 1e-10)
    I_levels = len(np.unique(np.round(I_normalized * 2) / 2))  # Redondear a 0.5
    Q_levels = len(np.unique(np.round(Q_normalized * 2) / 2))
    is_16qam = (I_levels > 2) or (Q_levels > 2)  # Si hay más de 2 niveles → 16-QAM
    
    if is_bpsk:
        modulation_type = 'BPSK'
        region_colors = region_colors_bpsk
        region_names = region_names_bpsk
        num_regions = 2
    elif is_16qam:
        modulation_type = '16-QAM'
        region_colors = region_colors_16qam
        region_names = region_names_16qam
        num_regions = 16
    else:
        modulation_type = 'QPSK'
        region_colors = region_colors_qpsk
        region_names = region_names_qpsk
        num_regions = 4

    fig = go.Figure()

    if trace_symbols:
        # MODO TRAYECTORIAS: Seguir símbolos individuales
        num_symbols = slices[0][0].size
        z_positions = np.array([z_km for _, z_km in slices])
        
        # Calcular límites para ejes
        all_symbols = np.concatenate([s for s, _ in slices])
        xlim = lim_rob(all_symbols.real)
        ylim = lim_rob(all_symbols.imag)
        zmin, zmax = float(np.min(z_positions)), float(np.max(z_positions))
        
        if group_by_quadrant:
            # Agrupar símbolos por región inicial (adaptativo a BPSK/QPSK/16-QAM)
            initial_symbols = slices[0][0]
            regions = [assign_region_adaptive(sym, modulation_type) for sym in initial_symbols]
            
            # Crear trazas agrupadas por región
            for r in range(num_regions):
                # Índices de símbolos en esta región
                indices = [i for i, reg in enumerate(regions) if reg == r]
                if not indices:
                    continue
                
                # Trayectorias de todos los símbolos de esta región
                for sym_idx in indices:
                    traj_I = [s[sym_idx].real for s, _ in slices]
                    traj_Q = [s[sym_idx].imag for s, _ in slices]
                    traj_Z = z_positions.tolist()
                    
                    # CAMBIO: Z ahora es X (horizontal), I es Y, Q es Z (vertical)
                    fig.add_trace(go.Scatter3d(
                        name=region_names[r],
                        x=traj_Z,  # Distancia en eje horizontal
                        y=traj_I,  # In-Phase
                        z=traj_Q,  # Quadrature en vertical
                        mode="lines",
                        line=dict(width=6, color=region_colors[r]),  # Líneas más gruesas
                        opacity=0.7,
                        hovertemplate=f"{region_names[r]}<br>Distancia=%{{x:.1f}} km<br>I=%{{y:.4f}}<br>Q=%{{z:.4f}}<extra></extra>",
                        showlegend=(sym_idx == indices[0]),  # Solo mostrar una vez por región
                        legendgroup=f"r{r}",
                    ))
        else:
            # Modo original: color basado en fase inicial
            for sym_idx in range(num_symbols):
                # Extraer la trayectoria del símbolo sym_idx a través de todos los slices
                traj_I = [s[sym_idx].real for s, _ in slices]
                traj_Q = [s[sym_idx].imag for s, _ in slices]
                traj_Z = z_positions.tolist()
                
                # Color basado en la posición inicial del símbolo (para identificarlo)
                initial_phase = np.angle(slices[0][0][sym_idx])
                color_val = (initial_phase + np.pi) / (2 * np.pi)  # Normalizar 0-1
                
                # CAMBIO: Z ahora es X (horizontal), I es Y, Q es Z (vertical)
                fig.add_trace(go.Scatter3d(
                    name=f"Símbolo {sym_idx+1}" if num_symbols <= 10 else None,
                    x=traj_Z,  # Distancia en eje horizontal
                    y=traj_I,  # In-Phase
                    z=traj_Q,  # Quadrature en vertical
                    mode="lines",
                    line=dict(width=6, color=f"hsl({color_val*360:.0f}, 70%, 50%)"),  # Líneas más gruesas
                    hovertemplate=f"Símbolo {sym_idx+1}<br>Distancia=%{{x:.1f}} km<br>I=%{{y:.4f}}<br>Q=%{{z:.4f}}<extra></extra>",
                    showlegend=(num_symbols <= 10),
                ))
        
        # Añadir marcadores en las posiciones inicial y final
        initial_syms = slices[0][0]
        final_syms = slices[-1][0]
        
        # CAMBIO: Z es X (horizontal)
        fig.add_trace(go.Scatter3d(
            name="Constelación Tx",
            x=np.full(num_symbols, z_positions[0]),  # Distancia inicial
            y=initial_syms.real,  # I
            z=initial_syms.imag,  # Q
            mode="markers",
            marker=dict(size=8, color="lime", symbol="diamond", opacity=1.0,
                       line=dict(width=2, color="white")),
            hovertemplate="Tx<br>I=%{y:.4f}<br>Q=%{z:.4f}<extra></extra>",
            showlegend=True,
        ))
        
        fig.add_trace(go.Scatter3d(
            name="Constelación Rx",
            x=np.full(num_symbols, z_positions[-1]),  # Distancia final
            y=final_syms.real,  # I
            z=final_syms.imag,  # Q
            mode="markers",
            marker=dict(size=8, color="red", symbol="diamond", opacity=1.0,
                       line=dict(width=2, color="white")),
            hovertemplate="Rx<br>I=%{y:.4f}<br>Q=%{z:.4f}<extra></extra>",
            showlegend=True,
        ))
        
        # Añadir planos de corte en posiciones clave (solo planos, sin puntos)
        if show_slice_planes and len(z_positions) > 2:
            # Planos en: inicio, 25%, 50%, 75%, final
            key_positions = [0, len(z_positions)//4, len(z_positions)//2, 
                           3*len(z_positions)//4, len(z_positions)-1]
            
            for idx in key_positions:
                if idx >= len(slices):
                    continue
                z_km = z_positions[idx]
                
                # Crear malla para el plano (ahora perpendicular al eje X)
                I_range = np.linspace(ylim[0], ylim[1], 2)
                Q_range = np.linspace(xlim[0], xlim[1], 2)
                I_mesh, Q_mesh = np.meshgrid(I_range, Q_range)
                Z_mesh = np.full_like(I_mesh, z_km)
                
                # Color del plano basado en la posición
                plane_color = f'rgba(100, 100, 100, 0.1)'
                
                # CAMBIO: Plano perpendicular a X (distancia)
                fig.add_trace(go.Surface(
                    x=Z_mesh,  # Distancia (constante en el plano)
                    y=I_mesh,  # I
                    z=Q_mesh,  # Q
                    opacity=0.15,
                    colorscale=[[0, plane_color], [1, plane_color]],
                    showscale=False,
                    hoverinfo='skip',
                    name=f"Plano {z_km:.0f} km",
                    showlegend=False,
                ))
        
    else:
        # MODO NUBE TRADICIONAL
        X = np.concatenate([s.real for s, _ in slices])
        Y = np.concatenate([s.imag for s, _ in slices])
        Z = np.concatenate([np.full(s.shape, z_km) for s, z_km in slices])
        
        xlim = lim_rob(X)
        ylim = lim_rob(Y)
        zmin, zmax = float(np.min(Z)), float(np.max(Z))

        # 0) nube global
        fig.add_trace(go.Scatter3d(
            name="Nube",
            x=X, y=Y, z=Z,
            mode="markers",
            marker=dict(size=marker_size, color=Z, colorscale="Turbo", opacity=0.8,
                        colorbar=dict(title="Distancia (km)")),
            hovertemplate="I=%{x:.4f}<br>Q=%{y:.4f}<br>Z=%{z:.1f} km<extra></extra>",
            showlegend=True,
        ))

        # 1) centroides (opcional)
        if add_centroids:
            cx, cy, cz = [], [], []
            for s, z_km in slices:
                cx.append(np.mean(s.real))
                cy.append(np.mean(s.imag))
                cz.append(z_km)
            fig.add_trace(go.Scatter3d(
                name="Centroides",
                x=cx, y=cy, z=cz,
                mode="lines+markers",
                line=dict(width=4),
                marker=dict(size=3, color=cz, colorscale="Turbo"),
                hovertemplate="⟨I⟩=%{x:.4f}<br>⟨Q⟩=%{y:.4f}<br>Z=%{z:.1f} km<extra></extra>",
                showlegend=True,
            ))

        # 2) traza del slice actual (se animará)
        fig.add_trace(go.Scatter3d(
            name="Slice actual",
            x=[], y=[], z=[],
            mode="markers",
            marker=dict(size=3, color="white", opacity=0.9),
            hovertemplate="I=%{x:.4f}<br>Q=%{y:.4f}<br>Z=%{z:.1f} km<extra></extra>",
            showlegend=True,
        ))

    # índice de la traza que se actualizará en cada frame:
    slice_trace_idx = 2 if add_centroids else 1

    # frames + slider (solo en modo nube, no en modo trayectorias)
    if add_slider and not trace_symbols:
        frames = []
        steps = []
        for i, (s, z_km) in enumerate(slices):
            steps.append(dict(
                method="animate",
                label=f"{z_km:.0f} km",
                args=[[str(i)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}],
            ))
            frames.append(go.Frame(
                name=str(i),
                data=[go.Scatter3d(
                    x=s.real, y=s.imag, z=np.full(s.shape, z_km),
                    mode="markers", marker=dict(size=3, color="white"),
                    showlegend=False,
                )],
                traces=[slice_trace_idx],
            ))
        fig.frames = frames

        fig.update_layout(
            updatemenus=[dict(
                type="buttons", showactive=False, y=1.18, x=0.0,
                buttons=[
                    dict(label="▶ Play", method="animate",
                         args=[None, {"fromcurrent": True, "frame": {"duration": 300, "redraw": True}, "transition": {"duration": 0}}]),
                    dict(label="⏸ Pause", method="animate",
                         args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}]),
                ],
            )],
            sliders=[dict(active=0, y=1.05, len=0.9, pad=dict(t=10, b=10), steps=steps)]
        )

    # Calcular límites para el layout
    if trace_symbols:
        all_symbols = np.concatenate([s for s, _ in slices])
        xlim = lim_rob(all_symbols.real)  # Límites para I
        ylim = lim_rob(all_symbols.imag)  # Límites para Q
        z_positions = np.array([z_km for _, z_km in slices])
        zmin, zmax = float(np.min(z_positions)), float(np.max(z_positions))
    
    title_text = "Evolución 3D de constelaciones a lo largo del enlace"

    # CAMBIO: Ejes reordenados - X es distancia (horizontal), Y es I, Z es Q (vertical)
    fig.update_layout(
        template=theme,
        scene=dict(
            xaxis_title="Distancia (km)",  # Ahora horizontal
            yaxis_title="In-Phase (I)", 
            zaxis_title="Quadrature (Q)",  # Ahora vertical
            xaxis=dict(range=[zmin, zmax]),  # Distancia en X
            yaxis=dict(range=xlim),  # I en Y
            zaxis=dict(range=ylim),  # Q en Z
            aspectmode="manual",
            aspectratio=dict(x=5, y=1, z=1),  # Mucho más ancho en X (distancia)
        ),
        margin=dict(l=0, r=0, t=80, b=0),
        height=760,
        title=dict(
            text=title_text,
            y=0.98,  # Posición vertical del título (cerca del borde superior)
            x=0.5,   # Centrado horizontalmente
            xanchor='center',
            yanchor='top'
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        uirevision=True,  # mantiene zoom/rotación al mover el slider
    )
    fig.write_html(str(outpath), include_plotlyjs="cdn")



