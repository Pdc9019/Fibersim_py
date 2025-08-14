from __future__ import annotations
from typing import Sequence
import matplotlib.pyplot as plt
from pathlib import Path
from .array_api import xp
from .array_api import asnumpy as _asnp

def save_constellations_grid(consSym: Sequence, consZ_m: Sequence[float], outpath: Path):
    n = len(consSym)
    if n == 0:
        return
    rows = int((n - 1) ** 0.5) + 1
    cols = rows
    fig = plt.figure(figsize=(cols * 3.2, rows * 3.2))
    for i, (sym, z) in enumerate(zip(consSym, consZ_m), start=1):
        ax = fig.add_subplot(rows, cols, i)
        s = _asnp(sym)
        ax.plot(s.real, s.imag, ".", markersize=1)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True)
        ax.set_title(f"{z/1e3:.0f} km")
    fig.suptitle("Evolución de constelación (post-Rx)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
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
    fig.savefig(outpath, dpi=120)
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
    fig.savefig(outpath, dpi=120)
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
    fig.savefig(outpath, dpi=140)
    plt.close(fig)

def save_constellations_3d_html(consSym, consZ_m, outpath: Path,
                                every: int = 1, pts_per_slice: int = 2000, marker_size: float = 2.0,
                                theme: str = "plotly_dark",
                                add_centroids: bool = False,
                                add_slider: bool = True):
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
        if s.size > pts_per_slice:
            sel = np.random.choice(s.size, size=pts_per_slice, replace=False)
            s = s[sel]
        slices.append((s, float(z_m) / 1e3))  # km

    if not slices:
        return

    X = np.concatenate([s.real for s, _ in slices])
    Y = np.concatenate([s.imag for s, _ in slices])
    Z = np.concatenate([np.full(s.shape, z_km) for s, z_km in slices])

    def lim_rob(a):
        m = float(np.percentile(a, 99.5))
        return (-m, m)
    xlim = lim_rob(X)
    ylim = lim_rob(Y)
    zmin, zmax = float(np.min(Z)), float(np.max(Z))

    fig = go.Figure()

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

    # frames + slider (sin None en data, usamos 'traces=[idx]')
    if add_slider:
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

    fig.update_layout(
        template=theme,
        scene=dict(
            xaxis_title="In-Phase", yaxis_title="Quadrature", zaxis_title="Distancia (km)",
            xaxis=dict(range=xlim), yaxis=dict(range=ylim),
            zaxis=dict(range=[zmin, zmax]),
            aspectmode="cube",
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        height=760,
        title="Evolución 3D de constelaciones a lo largo del enlace",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        uirevision=True,  # mantiene zoom/rotación al mover el slider
    )
    fig.write_html(str(outpath), include_plotlyjs="cdn")



