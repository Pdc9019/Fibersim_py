from __future__ import annotations
from typing import List, Dict, Any
from pathlib import Path
import matplotlib.pyplot as plt

def save_profile_png(profile: List[Dict[str, Any]], out_png: Path) -> None:
    if not profile:
        return
    z = [p["z_km"] for p in profile]
    P = [p.get("P_dBm", None) for p in profile]
    OSNR = [p.get("OSNR_dB", None) for p in profile]
    BER = [p.get("BER", None) for p in profile]

    fig = plt.figure(figsize=(9, 5))
    ax1 = fig.add_subplot(111)
    ax1.plot(z, P, label="Potencia [dBm]")
    if any(v is not None for v in OSNR):
        ax1.plot(z, OSNR, label="OSNR [dB]")
    ax1.set_xlabel("Distancia [km]")
    ax1.set_ylabel("dB / dBm")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")
    # segundo eje para BER si existe
    if any(v is not None for v in BER):
        ax2 = ax1.twinx()
        ax2.plot(z, [v if v is not None else float("nan") for v in BER], label="BER", linestyle="--")
        ax2.set_yscale("log")
        ax2.set_ylabel("BER")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)

def save_profile_html(profile: List[Dict[str, Any]], out_html: Path) -> None:
    import plotly.graph_objs as go
    from plotly.offline import plot

    z = [p["z_km"] for p in profile]
    P = [p.get("P_dBm", None) for p in profile]
    OSNR = [p.get("OSNR_dB", None) for p in profile]
    BER = [p.get("BER", None) for p in profile]

    traces = [
        go.Scatter(x=z, y=P, mode="lines+markers", name="Potencia [dBm]"),
        go.Scatter(x=z, y=OSNR, mode="lines+markers", name="OSNR [dB]"),
    ]
    if any(v is not None for v in BER):
        traces.append(go.Scatter(x=z, y=BER, mode="lines+markers", name="BER", yaxis="y2"))

    layout = go.Layout(
        title="Perfiles z",
        xaxis=dict(title="Distancia [km]"),
        yaxis=dict(title="dB / dBm"),
        yaxis2=dict(title="BER", overlaying="y", side="right", type="log"),
        legend=dict(orientation="h"),
        template="plotly_white",
    )
    fig = go.Figure(data=traces, layout=layout)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    plot(fig, filename=str(out_html), auto_open=False, include_plotlyjs="cdn")
