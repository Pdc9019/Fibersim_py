from __future__ import annotations
import os, json, time, pathlib, importlib, math
from typing import Any, Dict
import typer
from rich import print as rprint

app = typer.Typer(help="Simulador de fibra con RRC + SSFM + plots 2D y 3D.")

# ------------------------- util -------------------------

def _load_config(config_path: str) -> Dict[str, Any]:
    p = pathlib.Path(config_path)
    return json.loads(p.read_text(encoding="utf-8"))

def _prepare_backend(use_gpu: bool):
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
        "fibersim.core.modem",
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
    modem = mods["fibersim.core.modem"]
    try:
        dsp = importlib.import_module("fibersim.core.dsp")
    except Exception:
        dsp = None

    try:
        backend_info = "CuPy"
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
    modem,
    dsp,
    )

def _to_numpy_if_needed(arr, xp):
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

# ------------------------- helpers métricas -------------------------

def _snr_sym_db(tx_ref_np, rx_syms_np) -> float:
    """SNR a nivel de símbolo tras alinear fase al MMSE."""
    import numpy as np
    n = min(len(tx_ref_np), len(rx_syms_np))
    if n == 0:
        return float("nan")
    tx = tx_ref_np[:n]
    rx = rx_syms_np[:n]
    num = np.vdot(tx, rx)  # conj(tx) @ rx
    theta = float(np.angle(num))
    rx_rot = rx * np.exp(-1j * theta)
    err = rx_rot - tx
    ps = float(np.mean(np.abs(tx) ** 2))
    pn = float(np.mean(np.abs(err) ** 2))
    if pn <= 0:
        return float("inf")
    return 10.0 * math.log10(ps / pn)

# ------------------------- ejecución principal -------------------------

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
        modem,
        dsp,
    ) = _prepare_backend(gpu)

    cfg = _load_config(config)
    parGlob = cfg["global"]
    chain = cfg["chain"]
    pulse_par = cfg["pulse"]
    dsp_par = cfg.get("dsp", {})

    # TX
    info: Dict[str, Any] = {}
    mod = str(parGlob.get("mod", "BPSK")).upper()
    rx_mode = str(parGlob.get("rx", "imdd"))
    # Choose PRBS M based on modulation to keep Nsym symbols consistent
    M_tx = {"BPSK": 2, "QPSK": 4, "16QAM": 16}.get(mod, 2)
    bits, info = prbs_gen(parGlob["Nsym"], M_tx, info)
    # Map bits to symbols on selected backend
    syms = modem.map_bits_to_symbols(bits, M_tx, xp)

    pulse_par = dict(pulse_par)
    pulse_par["Rb"] = parGlob["Rb"]
    pulse_par["Fs"] = parGlob["Fs"]

    txSig, info = pulse_shaper(syms, info, pulse_par)
    Ein = xp.sqrt(parGlob["Ptx"]) * txSig

    # Cadena
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

    # Rutas de salida
    outdir_p = pathlib.Path(outdir); outdir_p.mkdir(parents=True, exist_ok=True)
    plots_p = pathlib.Path(plots_dir); plots_p.mkdir(parents=True, exist_ok=True)
    log_name = f"simlog_{time.strftime('%Y-%m-%d_%H-%M-%S')}.json"

    # ---------------- RX / métricas ----------------
    sps = int(diag.get("sps", parGlob["sps"]))
    span = int(pulse_par.get("span", 8))
    # delay_samp in diag already accounts for TX+RX filter group delay (≈ span*sps)
    delay_guess = int(diag.get("delay_samp", 0))

    Aout_np = _to_numpy_if_needed(Aout, xp)
    syms_np = _to_numpy_if_needed(syms, xp)
    Nsym = int(parGlob["Nsym"])

    # Branch: BPSK IMDD keeps best-delay search; otherwise deterministic slicing + alignment
    EVM_dB = None; Q_post = None
    if (M_tx == 2) and (str(rx_mode).lower() == "imdd"):
        best_delay, BER_est, s_hat = modem.find_best_delay(
            rx_wave=Aout_np,
            sps=sps,
            tx_syms_ref=syms_np[:Nsym],
            guess_delay=delay_guess,
            halfwin=max(8, sps * 2),
        )
        delay_total = int(best_delay)
    else:
        # Sample to symbols deterministically around measured delay
        s_hat = modem.slice_to_symbols(Aout_np, sps=sps, delay_samp=delay_guess, Nsym=Nsym)
        delay_total = int(delay_guess)
        # For QPSK/16QAM normalize and align phase vs tx reference
        if M_tx > 2:
            s_hat = modem.normalize_constellation(s_hat, None)
            # tx reference already has unit power; ensure same truncation length
            tx_ref = syms_np[: len(s_hat)]
            s_hat = modem.carrier_phase_align(s_hat, tx_ref)
        # Metrics
        BER_est = modem.ber_from_symbols(syms_np[: len(s_hat)], s_hat, M=M_tx)
        try:
            EVM_dB = modem.evm_rms_db(syms_np[: len(s_hat)], s_hat)
        except Exception:
            EVM_dB = None
        try:
            evm_lin = 10 ** (float(EVM_dB) / 20.0) if EVM_dB is not None else None
            Q_post = modem.q_factor_from_evm(evm_lin) if evm_lin is not None else None
        except Exception:
            Q_post = None
    try:
        snr_sym_db = _snr_sym_db(syms_np[: len(s_hat)], s_hat)
    except Exception:
        snr_sym_db = float("nan")

    # Perfil medido (si hay)
    osnr_final_db = None
    profile = None
    try:
        powZ = diag.get("powZ_m", None)
        powW = diag.get("powW_W", None)
        osnrZ = diag.get("osnrZ_dB", None)
        if powZ is not None and powW is not None and len(powZ) == len(powW):
            profile = []
            for i, (z, p) in enumerate(zip(powZ, powW)):
                z_km = float(z) / 1e3
                try:
                    p_dbm = 10.0 * math.log10(max(float(p), 1e-30) / 1e-3)
                except Exception:
                    p_dbm = None
                osnr_i = None
                if osnrZ is not None and i < len(osnrZ) and osnrZ[i] is not None:
                    try:
                        osnr_i = float(osnrZ[i])
                    except Exception:
                        osnr_i = None
                profile.append({"z_km": z_km, "P_dBm": p_dbm, "OSNR_dB": osnr_i})
            # último OSNR válido
            if osnrZ:
                for v in reversed(osnrZ):
                    if v is not None:
                        osnr_final_db = float(v)
                        break
    except Exception:
        profile = None

    # Pout dBm si vino Pmean
    pout_dbm = None
    if "Pmean" in info:
        try:
            pout_dbm = 10.0 * math.log10(max(float(info["Pmean"]), 1e-30) / 1e-3)
        except Exception:
            pout_dbm = None

    result = {
        "status": "ok",
        "notes": "RRC+SSFM; snapshots guardados si do_const=True",
        "Lcum_m": info.get("Lcum", 0.0),
        "G_dB": info.get("G_dB", 0.0),
        "Pmean_W": info.get("Pmean", None),
        "backend": backend_info,
        "elapsed_s": elapsed,
        "delay_best_samp": delay_total,
        "BER_est_BPSK": None if M_tx > 2 else BER_est,
        "BER_post": BER_est if M_tx > 2 else None,
        "EVM_post_dB": EVM_dB if M_tx > 2 else None,
        "Q_post": Q_post if M_tx > 2 else None,
        "SNR_sym_dB": snr_sym_db,
        "OSNR_final_dB": osnr_final_db,
        "Pout_dBm": pout_dbm,
        "profile": profile,
    }

    from .io import write_simlog
    write_simlog(outdir_p / log_name, cfg, result, elapsed)

    # Plots
    if do_const and len(diag.get("consSym", [])) > 0:
        consSym_np = _to_numpy_if_needed(diag["consSym"], xp)
        consZ_np = _to_numpy_if_needed(diag["consZ_m"], xp)
        powZ_np = _to_numpy_if_needed(diag.get("powZ_m", []), xp)
        powW_np = _to_numpy_if_needed(diag.get("powW_W", []), xp)
        # Normalizar cada slice para visualización (no afecta métricas)
        try:
            consSym_np = [modem.normalize_constellation(c) for c in consSym_np]
        except Exception:
            pass

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
        save_eyediagram(Aout_np, sps, delay_total, plots_p / "eye.png")

    rprint(f"[bold cyan]{backend_info}[/bold cyan]")
    rprint(f"[bold green]Listo[/bold green]: log en [cyan]{outdir}/{log_name}[/cyan], "
           f"plots en [cyan]{plots_dir}[/cyan].")
    ber_print = result.get("BER_est_BPSK")
    if ber_print is None:
        ber_print = result.get("BER_post")
    ber_str = (f"{ber_print:.3e}" if isinstance(ber_print, (int, float)) and ber_print is not None else "n/a")
    rprint(f"L = {result['Lcum_m']/1e3:.1f} km | G = {result['G_dB']:.1f} dB | elapsed = {elapsed:.3f} s "
           f"| BER={ber_str} | SNRsym={snr_sym_db:.2f} dB"
           + (f" | OSNR={osnr_final_db:.2f} dB" if osnr_final_db is not None else ""))

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
    const3d_pts: int = typer.Option(1000, help="Máx. puntos por snapshot para PNG 3D."),
    do_const3d_html: bool = typer.Option(True, help="Guardar 3D interactivo HTML con Plotly."),
    const3d_html_pts: int = typer.Option(1200, help="Máx. puntos por snapshot para HTML 3D."),
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
