from __future__ import annotations
import os, json, time, pathlib, importlib, math
from typing import Any, Dict, List
import typer
from rich import print as rprint

app = typer.Typer(help="Simulador de fibra con RRC + SSFM + plots 2D y 3D.")

# ------------------------- util -------------------------

def _load_config(config_path: str) -> Dict[str, Any]:
    p = pathlib.Path(config_path)
    return json.loads(p.read_text(encoding="utf-8"))

def _prepare_backend(use_gpu: bool):
    """
    Fija la variable de entorno y recarga array_api para elegir xp.
    Luego recarga módulos que puedan haber capturado xp al importar.
    Devuelve (xp, backend_info) y referencias a las funciones del solver.
    """
    os.environ["FIBERSIM_GPU"] = "1" if use_gpu else "0"

    from .core import array_api as _array_api
    _array_api = importlib.reload(_array_api)
    xp = _array_api.xp

    mod_names = [
        "fibersim.core.prbs",
        "fibersim.core.pulse",
        "fibersim.core.chain",
        "fibersim.core.plot",
        "fibersim.core.utils",
        "fibersim.core.fiber",
    ]
    mods = {}
    for name in mod_names:
        m = importlib.import_module(name)
        mods[name] = importlib.reload(m)

    prbs_gen = mods["fibersim.core.prbs"].prbs_gen
    pulse_shaper = mods["fibersim.core.pulse"].pulse_shaper
    run_chain = mods["fibersim.core.chain"].run_chain
    save_constellations_grid = mods["fibersim.core.plot"].save_constellations_grid
    save_eyediagram = mods["fibersim.core.plot"].save_eyediagram
    save_power_evolution = mods["fibersim.core.plot"].save_power_evolution
    save_constellations_3d = mods["fibersim.core.plot"].save_constellations_3d
    save_constellations_3d_html = mods["fibersim.core.plot"].save_constellations_3d_html

    try:
        if getattr(xp, "__name__", "") == "cupy":
            dev_id = xp.cuda.runtime.getDevice()
            props = xp.cuda.runtime.getDeviceProperties(dev_id)
            name = props.get("name", b"").decode(errors="ignore") if isinstance(props.get("name", b""), (bytes, bytearray)) else props.get("name", "GPU")
            backend_info = f"GPU CuPy - {name}"
        else:
            backend_info = "CPU NumPy"
    except Exception:
        backend_info = "CPU NumPy"

    return (
        xp,
        backend_info,
        prbs_gen,
        pulse_shaper,
        run_chain,
        save_constellations_grid,
        save_eyediagram,
        save_power_evolution,
        save_constellations_3d,
        save_constellations_3d_html,
    )

def _to_numpy_if_needed(arr, xp):
    """
    Convierte a numpy si el backend es CuPy. Si ya es numpy o lista de arrays, maneja ambos casos.
    """
    try:
        asnumpy = getattr(xp, "asnumpy", None)
        if asnumpy is None:
            return arr
        if isinstance(arr, (list, tuple)):
            return type(arr)([asnumpy(a) for a in arr])
        if isinstance(arr, dict):
            return {k: _to_numpy_if_needed(v, xp) for k, v in arr.items()}
        return asnumpy(arr)
    except Exception:
        return arr

# ------------------------- perfil analitico Potencia, OSNR, BER -------------------------

_h = 6.62607015e-34
_c = 299792458.0

def _W_to_dBm(p_W: float) -> float:
    return 10.0 * math.log10(max(p_W, 1e-30) / 1e-3)

def _build_power_osnr_profile(cfg: Dict[str, Any], Bo_Hz: float = 12.5e9) -> List[Dict[str, Any]]:
    """
    Recorre la chain aplicando atenuacion de fibra y ganancia de EDFA, acumulando ASE.
    Devuelve lista de puntos con z_km, P_dBm y OSNR_dB cuando corresponda.
    Usa nsp del bloque EDFA si esta presente, si no usa 2.5 por defecto.
    """
    g = cfg["global"]
    chain = cfg["chain"]
    lambda_nm = float(g.get("lambda_nm", 1550.0))
    nu = _c / (lambda_nm * 1e-9)

    z_m = 0.0
    P_sig_W = float(g["Ptx"])
    P_ase_W = 0.0

    prof: List[Dict[str, Any]] = []
    for i, blk in enumerate(chain):
        t = blk["type"]
        par = blk["par"]

        if t == "fiber":
            L = float(par["L"])
            alpha = float(par["alpha"])  # 1/m sobre potencia
            P_sig_W *= math.exp(-alpha * L)
            z_m += L
            prof.append({
                "i": i, "kind": "fiber", "z_km": z_m/1e3,
                "P_dBm": _W_to_dBm(P_sig_W),
                "OSNR_dB": None if P_ase_W <= 0 else 10*math.log10(P_sig_W / max(P_ase_W, 1e-30)),
            })

        elif t == "edfa":
            G_dB = float(par.get("G_dB", 0.0))
            G_lin = 10.0**(G_dB/10.0)
            nsp = float(par.get("nsp", 2.5))

            # amplificacion de señal
            P_sig_W *= G_lin
            # ruido ASE de doble polarizacion en Bo
            P_ase_W += 2.0 * nsp * _h * nu * (G_lin - 1.0) * Bo_Hz

            prof.append({
                "i": i, "kind": "edfa", "z_km": z_m/1e3,
                "P_dBm": _W_to_dBm(P_sig_W),
                "OSNR_dB": 10*math.log10(P_sig_W / max(P_ase_W, 1e-30)),
                "G_dB": G_dB, "nsp": nsp,
            })
        else:
            prof.append({
                "i": i, "kind": t, "z_km": z_m/1e3,
                "P_dBm": _W_to_dBm(P_sig_W),
                "OSNR_dB": None if P_ase_W <= 0 else 10*math.log10(P_sig_W / max(P_ase_W, 1e-30)),
            })
    return prof

def _add_ber_to_profile(profile: List[Dict[str, Any]], Rb: float, M: int) -> None:
    """
    Agrega BER aproximado por punto a partir de OSNR usando mapeo SNR_elec ≈ OSNR_lin*(Bo/Rb).
    Para M distinto de 2 se usa una aproximacion burda.
    """
    from math import erfc, sqrt
    Bo_Hz = 12.5e9
    for pt in profile:
        osnr = pt.get("OSNR_dB", None)
        if osnr is None:
            pt["BER"] = None
            continue
        OSNR_lin = 10.0**(osnr/10.0)
        SNR_lin = OSNR_lin * (Bo_Hz / max(Rb, 1.0))
        if M == 2:
            BER = 0.5 * erfc(sqrt(max(SNR_lin, 1e-12))/sqrt(2.0))
        elif M == 4:
            BER = 0.5 * erfc(sqrt(0.5*max(SNR_lin, 1e-12)))
        elif M == 16:
            BER = (0.75/4.0) * erfc(sqrt(0.1*max(SNR_lin, 1e-12)))
        else:
            BER = 0.2 * erfc(sqrt(0.1*max(SNR_lin, 1e-12)))
        pt["BER"] = BER

def _save_profile_png(profile: List[Dict[str, Any]], out_png: pathlib.Path) -> None:
    if not profile:
        return
    import matplotlib.pyplot as plt
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
    if any(v is not None for v in BER):
        ax2 = ax1.twinx()
        ax2.plot(z, [v if v is not None else float("nan") for v in BER], label="BER", linestyle="--")
        ax2.set_yscale("log")
        ax2.set_ylabel("BER")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)

def _save_profile_html(profile: List[Dict[str, Any]], out_html: pathlib.Path) -> None:
    try:
        import plotly.graph_objs as go
        from plotly.offline import plot
    except Exception:
        return  # si falta plotly, omitimos HTML

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

# ------------------------- ejecucion principal -------------------------

def _execute(
    config: str,
    outdir: str,
    gpu: bool,
    dz: float | None,
    insertion_db: float,
    use_insertion_loss: bool,
    splice_db: float,
    use_splice_loss: bool,
    do_const: bool,
    step_const_km: float,
    do_eye: bool,
    plots_dir: str,
    # opciones 3D
    do_const3d: bool,
    const3d_every: int,
    const3d_pts: int,
    do_const3d_html: bool,
    const3d_html_pts: int,
):
    (
        xp,
        backend_info,
        prbs_gen,
        pulse_shaper,
        run_chain,
        save_constellations_grid,
        save_eyediagram,
        save_power_evolution,
        save_constellations_3d,
        save_constellations_3d_html,
    ) = _prepare_backend(gpu)

    cfg = _load_config(config)
    parGlob = cfg["global"]
    chain = cfg["chain"]
    pulse_par = cfg["pulse"]

    # ---------- TX ----------
    info: Dict[str, Any] = {}
    bits, info = prbs_gen(parGlob["Nsym"], parGlob["M"], info)
    syms = (1 - 2 * bits.astype(xp.int8)).astype(xp.complex128)

    import numpy as _np
    syms_tx_cpu = _to_numpy_if_needed(syms, xp)

    pulse_par = dict(pulse_par)
    pulse_par["Rb"] = parGlob["Rb"]
    pulse_par["Fs"] = parGlob["Fs"]

    txSig, info = pulse_shaper(syms, info, pulse_par)
    Ein = xp.sqrt(parGlob["Ptx"]) * txSig

    # ---------- Cadena ----------
    t0 = time.time()
    Aout, info, diag = run_chain(
        Ein,
        info,
        chain,
        parGlob,
        dz_override=dz,
        use_insertion_loss=use_insertion_loss,
        insertion_dB=insertion_db,
        use_splice_loss=use_splice_loss,
        splice_dB=splice_db,
        do_const=do_const,
        step_const_m=step_const_km * 1e3,
    )
    elapsed = time.time() - t0

    # ---------- BER BPSK robusto ----------
    ber_est = None
    try:
        sps = int(diag.get("sps", int(parGlob["sps"])))
        delay_tx = int(diag.get("delay_samp", 0))
        from .core.utils import get_rx_filter
        roll = float(pulse_par.get("roll", 0.1))
        span = int(pulse_par.get("span", 10))
        rx_filt = get_rx_filter(sps=sps, roll=roll, span=span)

        y = rx_filt(Aout)
        y_np = _to_numpy_if_needed(y, xp)

        h_len = span * sps + 1
        rx_gd = (h_len - 1) // 2

        start = max(0, delay_tx + rx_gd)
        n_avail = max(0, len(y_np) - start)
        Nsym = int(parGlob["Nsym"])
        n_syms = max(0, min(Nsym, n_avail // sps))

        y_s = y_np[start : start + n_syms * sps : sps]
        s_tx = syms_tx_cpu[:n_syms]

        if n_syms > 0:
            s_hat = _np.where(y_s.real >= 0, 1.0, -1.0).astype(_np.complex128)
            b_tx = (s_tx.real < 0).astype(_np.uint8)
            b_rx = (s_hat.real < 0).astype(_np.uint8)
            n = min(len(b_tx), len(b_rx))
            ber_est = float(_np.mean(b_tx[:n] ^ b_rx[:n]))
    except Exception:
        ber_est = None

    # ---------- Logs + Plots ----------
    outdir_p = pathlib.Path(outdir); outdir_p.mkdir(parents=True, exist_ok=True)
    plots_p = pathlib.Path(plots_dir); plots_p.mkdir(parents=True, exist_ok=True)
    log_name = f"simlog_{time.strftime('%Y-%m-%d_%H-%M-%S')}.json"

    # Plots existentes
    if do_const and len(diag.get("consSym", [])) > 0:
        consSym_np = _to_numpy_if_needed(diag["consSym"], xp)
        consZ_np = _to_numpy_if_needed(diag["consZ_m"], xp)
        powZ_np = _to_numpy_if_needed(diag.get("powZ_m", []), xp)
        powW_np = _to_numpy_if_needed(diag.get("powW_W", []), xp)

        save_constellations_grid(consSym_np, consZ_np, plots_p / "constelaciones.png")
        save_power_evolution(powZ_np, powW_np, plots_p / "potencia.png", unit="dBm")

        if do_const3d:
            save_constellations_3d(
                consSym_np, consZ_np,
                plots_p / "constelaciones_3d.png",
                every=const3d_every, pts_per_slice=const3d_pts, marker_size=1.0
            )
        if do_const3d_html:
            save_constellations_3d_html(
                consSym_np, consZ_np,
                plots_p / "constelaciones_3d.html",
                every=const3d_every, pts_per_slice=const3d_html_pts, marker_size=2.0
            )

    if do_eye:
        Aout_np = _to_numpy_if_needed(Aout, xp)
        save_eyediagram(Aout_np, diag["sps"], diag["delay_samp"], plots_p / "eye.png")

    # Perfil analitico y escritura de archivos que la GUI ya espera
    profile = _build_power_osnr_profile(cfg, Bo_Hz=12.5e9)
    _add_ber_to_profile(profile, Rb=float(parGlob["Rb"]), M=int(parGlob.get("M", 2)))
    _save_profile_html(profile, plots_p / "perfil.html")
    _save_profile_png(profile, plots_p / "perfil.png")

    # Resultado
    last_osnr = None
    for pt in reversed(profile):
        if pt.get("OSNR_dB", None) is not None:
            last_osnr = pt["OSNR_dB"]
            break

    result = {
        "status": "ok",
        "notes": "RRC+SSFM; snapshots si do_const=True; perfil analitico guardado",
        "Lcum_m": info.get("Lcum", 0.0),
        "G_dB": info.get("G_dB", 0.0),
        "Pmean_W": info.get("Pmean", None),
        "backend": backend_info,
        "elapsed_s": elapsed,
        "BER_est_BPSK": ber_est,
        "OSNR_final_dB": last_osnr,
        "Pout_dBm": profile[-1]["P_dBm"] if profile else None,
    }

    from .io import write_simlog
    write_simlog(outdir_p / log_name, cfg, result, elapsed)

    rprint(f"[bold cyan]{backend_info}[/bold cyan]")
    rprint(f"[bold green]Listo[/bold green]: log en [cyan]{outdir}/{log_name}[/cyan], plots en [cyan]{plots_dir}[/cyan].")
    if ber_est is not None:
        rprint(f"BER BPSK estimado = {ber_est:.3e}")
    if last_osnr is not None:
        rprint(f"OSNR final analitico = {last_osnr:.2f} dB")
    rprint(f"L = {result['Lcum_m']/1e3:.1f} km | G = {result['G_dB']:.1f} dB | elapsed = {elapsed:.3f} s")

    return backend_info

# ------------------------- CLI -------------------------

@app.command("run")
def run(
    config: str = typer.Argument(..., help="Ruta a archivo JSON de configuración."),
    outdir: str = typer.Option("logs", help="Carpeta para logs JSON."),
    gpu: bool = typer.Option(True, help="Usar GPU si está disponible."),
    dz: float | None = typer.Option(None, help="Override de dz global en metros."),
    insertion_db: float = typer.Option(1.0, help="Pérdida por inserción en dB."),
    use_insertion_loss: bool = typer.Option(True, help="Aplicar pérdida de inserción inicial."),
    splice_db: float = typer.Option(0.2, help="Pérdida por fusión entre fibras en dB."),
    use_splice_loss: bool = typer.Option(True, help="Aplicar pérdida por fusión entre tramos."),
    do_const: bool = typer.Option(True, help="Capturar constelaciones durante la propagación."),
    step_const_km: float = typer.Option(5.0, help="Paso entre capturas de constelación en km."),
    do_eye: bool = typer.Option(True, help="Guardar eye diagram al final."),
    plots_dir: str = typer.Option("plots", help="Carpeta para imágenes."),
    do_const3d: bool = typer.Option(False, help="Guardar PNG 3D con matplotlib."),
    const3d_every: int = typer.Option(1, help="Usar 1 de cada N snapshots en 3D PNG."),
    const3d_pts: int = typer.Option(1000, help="Máx. puntos por snapshot para 3D PNG."),
    do_const3d_html: bool = typer.Option(True, help="Guardar 3D interactivo HTML con Plotly."),
    const3d_html_pts: int = typer.Option(1200, help="Máx. puntos por snapshot para 3D HTML."),
):
    _execute(
        config=config,
        outdir=outdir,
        gpu=gpu,
        dz=dz,
        insertion_db=insertion_db,
        use_insertion_loss=use_insertion_loss,
        splice_db=splice_db,
        use_splice_loss=use_splice_loss,
        do_const=do_const,
        step_const_km=step_const_km,
        do_eye=do_eye,
        plots_dir=plots_dir,
        do_const3d=do_const3d,
        const3d_every=const3d_every,
        const3d_pts=const3d_pts,
        do_const3d_html=do_const3d_html,
        const3d_html_pts=const3d_html_pts,
    )

if __name__ == "__main__":
    app()
