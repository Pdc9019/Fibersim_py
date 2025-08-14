from __future__ import annotations
import os, json, time, pathlib
from typing import Any, Dict
import typer

from rich import print as rprint

from .core.prbs import prbs_gen
from .core.pulse import pulse_shaper
from .core.chain import run_chain
from .core.plot import (
    save_constellations_grid,
    save_eyediagram,
    save_power_evolution,
    save_constellations_3d,
    save_constellations_3d_html,  # << interactivo
)
from .io import write_simlog

app = typer.Typer(help="Simulador de fibra con RRC + SSFM + plots (2D/3D).")

def _load_config(config_path: str) -> Dict[str, Any]:
    p = pathlib.Path(config_path)
    return json.loads(p.read_text(encoding="utf-8"))

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
    # --- opciones 3D ---
    do_const3d: bool,
    const3d_every: int,
    const3d_pts: int,
    do_const3d_html: bool,
    const3d_html_pts: int,
):
    # Selección CPU/GPU antes de importar el backend
    os.environ["FIBERSIM_GPU"] = "1" if gpu else "0"
    from .core.array_api import xp  # importar tras fijar FIBERSIM_GPU

    # ---------- Cargar config ----------
    cfg = _load_config(config)
    parGlob = cfg["global"]
    chain = cfg["chain"]
    pulse_par = cfg["pulse"]

    # ---------- TX ----------
    info: Dict[str, Any] = {}
    bits, info = prbs_gen(parGlob["Nsym"], parGlob["M"], info)
    # BPSK
    syms = (1 - 2 * bits.astype(xp.int8)).astype(xp.complex128)

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

    # ---------- Logs + Plots ----------
    outdir_p = pathlib.Path(outdir); outdir_p.mkdir(parents=True, exist_ok=True)
    plots_p = pathlib.Path(plots_dir); plots_p.mkdir(parents=True, exist_ok=True)
    log_name = f"simlog_{time.strftime('%Y-%m-%d_%H-%M-%S')}.json"

    result = {
        "status": "ok",
        "notes": "RRC+SSFM; snapshots guardados si do_const=True",
        "Lcum_m": info.get("Lcum", 0.0),
        "G_dB": info.get("G_dB", 0.0),
        "Pmean_W": info.get("Pmean", None),
    }
    write_simlog(outdir_p / log_name, cfg, result, elapsed)

    if do_const and len(diag["consSym"]) > 0:
        # PNGs
        save_constellations_grid(diag["consSym"], diag["consZ_m"], plots_p / "constelaciones.png")
        save_power_evolution(diag["powZ_m"], diag["powW_W"], plots_p / "potencia.png", unit="dBm")
        if do_const3d:
            save_constellations_3d(
                diag["consSym"], diag["consZ_m"],
                plots_p / "constelaciones_3d.png",
                every=const3d_every, pts_per_slice=const3d_pts, marker_size=1.0
            )
        # HTML interactivo
        if do_const3d_html:
            save_constellations_3d_html(
                diag["consSym"], diag["consZ_m"],
                plots_p / "constelaciones_3d.html",
                every=const3d_every, pts_per_slice=const3d_html_pts, marker_size=2.0
            )
    if do_eye:
        save_eyediagram(Aout, diag["sps"], diag["delay_samp"], plots_p / "eye.png")

    rprint(f"[bold green]Listo[/bold green]: log en [cyan]{outdir}/{log_name}[/cyan], "
           f"plots en [cyan]{plots_dir}[/cyan].")
    rprint(f"L = {result['Lcum_m']/1e3:.1f} km | G = {result['G_dB']:.1f} dB | elapsed = {elapsed:.3f} s")

@app.command("run")
def run(
    config: str = typer.Argument(..., help="Ruta a archivo JSON de configuración."),
    outdir: str = typer.Option("logs", help="Carpeta para logs JSON."),
    gpu: bool = typer.Option(True, help="Usar GPU (si disponible)."),
    dz: float | None = typer.Option(None, help="Override de dz global (m)."),
    insertion_db: float = typer.Option(1.0, help="Pérdida por inserción (dB)."),
    use_insertion_loss: bool = typer.Option(True, help="Aplicar pérdida de inserción inicial."),
    splice_db: float = typer.Option(0.2, help="Pérdida por fusión entre fibras (dB)."),
    use_splice_loss: bool = typer.Option(True, help="Aplicar pérdida por fusión entre tramos contiguos."),
    do_const: bool = typer.Option(True, help="Capturar constelaciones durante la propagación."),
    step_const_km: float = typer.Option(5.0, help="Paso entre capturas de constelación (km)."),
    do_eye: bool = typer.Option(True, help="Guardar eye diagram al final."),
    plots_dir: str = typer.Option("plots", help="Carpeta para imágenes."),
    # --- opciones 3D ---
    do_const3d: bool = typer.Option(False, help="Guardar PNG 3D (matplotlib)."),
    const3d_every: int = typer.Option(1, help="Usar 1 de cada N snapshots en los 3D."),
    const3d_pts: int = typer.Option(1000, help="Máx. puntos por snapshot (PNG 3D)."),
    do_const3d_html: bool = typer.Option(True, help="Guardar 3D interactivo (HTML Plotly)."),
    const3d_html_pts: int = typer.Option(1200, help="Máx. puntos por snapshot (HTML 3D)."),
):
    _execute(
        config, outdir, gpu, dz, insertion_db, use_insertion_loss, splice_db, use_splice_loss,
        do_const, step_const_km, do_eye, plots_dir,
        do_const3d, const3d_every, const3d_pts,
        do_const3d_html, const3d_html_pts,
    )

if __name__ == "__main__":
    app()
