from __future__ import annotations
import os, json, time, pathlib, importlib, math
from typing import Any, Dict
import typer
from rich import print as rprint
import numpy as np

app = typer.Typer(help="Simulador de fibra con RRC + SSFM + plots 2D y 3D.")

# ------------------------- util -------------------------

def _load_config(config_path: str) -> Dict[str, Any]:
    p = pathlib.Path(config_path)
    return json.loads(p.read_text(encoding="utf-8"))

def _prepare_backend(use_gpu: bool):
    os.environ["FIBERSIM_GPU"] = "1" if use_gpu else "0"

    # Configurar backend ANTES que cualquier cálculo
    from .core import array_api as _array_api
    backend_str = _array_api.set_backend(use_gpu)
    
    # Recargar módulos después del cambio de backend
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
        "fibersim.core.awgn",
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
    awgn = mods["fibersim.core.awgn"]
    try:
        dsp = importlib.import_module("fibersim.core.dsp")
    except Exception:
        dsp = None

    try:
        if _array_api.backend_name == "cupy":
            try:
                dev_id = xp.cuda.runtime.getDevice()
                props = xp.cuda.runtime.getDeviceProperties(dev_id)
                name = props.get("name", b"").decode(errors="ignore") if isinstance(props.get("name", b""), (bytes, bytearray)) else props.get("name", "GPU")
                backend_info = f"GPU CuPy - {name}"
            except Exception:
                backend_info = "GPU CuPy"
        else:
            backend_info = backend_str
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
        awgn,
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
    """SNR a nivel de símbolo tras alinear fase al MMSE y aplicar DSP."""
    from .core.snr import calc_snr_post_dsp
    _, snr_db = calc_snr_post_dsp(rx_syms_np, tx_ref_np)
    return snr_db
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
    step_plot2d_km: float,
    do_eye: bool,
    plots_dir: str,
    do_const3d: bool,
    const3d_every: int,
    const3d_pts: int,
    do_const3d_html: bool,
    step_const3d_km: float,
    const3d_html_pts: int,
    trace_symbols: bool = True,
    num_traces: int = 50,
    group_by_quadrant: bool = True,
    show_slice_planes: bool = True,
    do_waveform: bool = False,
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
        awgn,
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

    # AWGN en TX (si está habilitado)
    P_awgn_tx = 0.0
    if parGlob.get("enable_awgn", False):
        snr_base = parGlob.get("awgn_intensity_db", 25.0)
        # TX: ruido ligeramente menor (mejor SNR) ya que es antes de propagación
        snr_tx = snr_base + 5.0
        print(f"[AWGN] Aplicando ruido TX: SNR={snr_tx:.1f} dB")
        Ein, P_awgn_tx = awgn.add_awgn(Ein, snr_tx, xp, sps=1, rolloff=0.0, mode="sample")
        info["P_AWGN_TX"] = P_awgn_tx
    else:
        print(f"[AWGN] DESACTIVADO")
        info["P_AWGN_TX"] = 0.0

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

    # AWGN en RX (si está habilitado)
    P_awgn_rx = 0.0
    if parGlob.get("enable_awgn", False):
        snr_base = parGlob.get("awgn_intensity_db", 25.0)
        # RX: usar intensidad base (ruido térmico + shot noise del receptor)
        print(f"[AWGN] Aplicando ruido RX: SNR={snr_base:.1f} dB")
        Aout, P_awgn_rx = awgn.add_awgn(Aout, snr_base, xp, sps=1, rolloff=0.0, mode="sample")
        info["P_AWGN_RX"] = P_awgn_rx
        
        # Capturar constelación POST-AWGN para visualización correcta del ruido RX
        if do_const and "consSym" in diag and len(diag.get("consSym", [])) > 0:
            # Crear RX filter (igual que en run_chain)
            from .core.utils import get_tx_filter
            h_np = get_tx_filter(sps=int(parGlob["sps"]), 
                                roll=pulse_par.get("roll", 0.1), 
                                span=int(pulse_par.get("span", 10)))
            h = xp.asarray(h_np, dtype=xp.float64)
            
            # Aplicar RX filter (convolución)
            den = xp.asarray([1.0], dtype=h.dtype)
            if hasattr(xp, 'convolve'):
                # NumPy o CuPy con convolve
                Arx_post_awgn = xp.convolve(Aout, h, mode="same")
            else:
                # Fallback a lfilter
                from cupyx.scipy import signal as xsignal
                Arx_post_awgn = xsignal.lfilter(h, den, Aout)
            
            # Extraer símbolos con AWGN RX incluido
            delay = 2 * info["pulseDelay"]
            syms_post_awgn = Arx_post_awgn[delay::int(parGlob["sps"])]
            
            # Reemplazar el último punto de constelación con versión post-AWGN
            diag["consSym"][-1] = syms_post_awgn
            
            print(f"[AWGN] Constelación final actualizada con ruido RX")
    else:
        print(f"[AWGN] (ruido solo de ASE de EDFAs)")
        info["P_AWGN_RX"] = 0.0

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
    
    # ============ COMPENSACIÓN DIGITAL DE DISPERSIÓN CROMÁTICA (CDC) ============
    # NOTA: Función chromatic_dispersion_compensation no implementada aún
    # La dispersión cromática debe compensarse mediante fibra DCF en la cadena
    beta2_total = 0.0
    for blk in chain:
        if blk.get("type") == "fiber":
            L = float(blk["par"].get("L", 0))  # metros
            beta2 = float(blk["par"].get("beta2", 0))  # s²/m
            beta2_total += beta2 * L
    
    if abs(beta2_total) > 1e-30:
        print(f"[CDC] Dispersión total acumulada: {beta2_total*1e24:.2f} ps²")
        print(f"[CDC] Nota: Compensación digital CDC no implementada. Use fibra DCF para compensar.")
    # ============================================================================

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
        
        # tx reference already has unit power; ensure same truncation length
        tx_ref = syms_np[: len(s_hat)]
        
        # For QPSK/16QAM: apply carrier phase alignment BEFORE normalization
        # BPSK NO necesita carrier_phase_align porque ber_from_symbols
        # ya hace su propia alineación considerando la ambigüedad de π
        if M_tx > 2:  # Solo para QPSK (4) y 16QAM (16)
            s_hat = modem.carrier_phase_align(s_hat, tx_ref)
        
        # IMPORTANTE: NO normalizar antes de calcular métricas
        # La normalización cancela el efecto del ruido en BER/EVM
        # Solo se debe usar para visualización de constelaciones
        
        # Metrics (calculadas con símbolos SIN normalizar)
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
    
    # SNR post-DSP eliminado - solo usamos OSNR para sistemas ópticos
    snr_sym_db = None

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
            # último OSNR válido de la cadena
            if osnrZ:
                for v in reversed(osnrZ):
                    if v is not None:
                        osnr_final_db = float(v)
                        break
            
            # OSNR mide solo el ruido óptico (ASE)
            # El AWGN TX viaja con la señal y no afecta al OSNR
            # El AWGN RX se añade después y degrada el SNR total pero no el OSNR
            # 
            # Para obtener SNR efectivo total en RX (incluyendo todos los ruidos):
            # 1/SNR_total = 1/OSNR + 1/SNR_AWGN_TX + 1/SNR_AWGN_RX
            # 
            # Pero esto es complicado porque AWGN TX ya está "quemado" en la señal
            # y se ha propagado. Por ahora reportamos solo OSNR óptico.
            # 
            # Para diagnóstico, calculamos SNR efectivo considerando AWGN RX:
            snr_eff_db = None
            if osnr_final_db is not None:
                P_awgn_rx = info.get("P_AWGN_RX", 0.0)
                P_sig_final = powW[-1] if powW else 1.0
                
                if P_awgn_rx > 1e-30:
                    # OSNR en lineal
                    osnr_lin = 10.0 ** (osnr_final_db / 10.0)
                    # SNR del AWGN RX
                    snr_awgn_rx_lin = P_sig_final / P_awgn_rx
                    # SNR efectivo total: 1/SNR_eff = 1/OSNR + 1/SNR_AWGN_RX
                    snr_eff_lin = 1.0 / (1.0/osnr_lin + 1.0/snr_awgn_rx_lin)
                    snr_eff_db = 10.0 * math.log10(max(snr_eff_lin, 1e-12))
                else:
                    # Sin AWGN RX, SNR efectivo = OSNR
                    snr_eff_db = osnr_final_db
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
        "SNR_eff_dB": snr_eff_db,  # SNR efectivo total (OSNR + AWGN)
        "Pout_dBm": pout_dbm,
        "profile": profile,
    }

    # Actualizar cfg para reflejar dz_override si se usó
    if dz is not None:
        cfg_copy = dict(cfg)
        cfg_copy["global"] = dict(cfg["global"])
        cfg_copy["global"]["dz_global_override"] = dz
        # También actualizar dz en cada bloque de fibra para que el log sea consistente
        cfg_copy["chain"] = []
        for blk in cfg["chain"]:
            blk_copy = dict(blk)
            if blk.get("type") == "fiber":
                blk_copy["par"] = dict(blk.get("par", {}))
                blk_copy["par"]["dz"] = dz
            cfg_copy["chain"].append(blk_copy)
    else:
        cfg_copy = cfg

    from .io import write_simlog
    write_simlog(outdir_p / log_name, cfg_copy, result, elapsed)

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

        # Filtrar datos para gráficos 2D según step_plot2d_km
        # step_const_km es el paso de captura, step_plot2d_km es el paso de graficado
        if len(consZ_np) > 1 and step_plot2d_km > step_const_km:
            z_diff_m = abs(consZ_np[1] - consZ_np[0]) if len(consZ_np) > 1 else step_const_km * 1000
            z_diff_km = z_diff_m / 1000.0
            plot2d_every = max(1, int(round(step_plot2d_km / z_diff_km)))
            
            # Diezmar pero SIEMPRE incluir el último punto (constelación final)
            consSym_plot2d = consSym_np[::plot2d_every]
            consZ_plot2d = consZ_np[::plot2d_every]
            
            # Forzar inclusión del punto final si no está ya incluido
            if consZ_plot2d[-1] != consZ_np[-1]:
                consSym_plot2d = list(consSym_plot2d) + [consSym_np[-1]]
                consZ_plot2d = list(consZ_plot2d) + [consZ_np[-1]]
        else:
            consSym_plot2d = consSym_np
            consZ_plot2d = consZ_np

        save_constellations_grid(consSym_plot2d, consZ_plot2d, plots_p / "constelaciones.png")
        save_power_evolution(powZ_np, powW_np, plots_p / "potencia.png", unit="dBm")

        # Calcular paso adaptativo para visualización 3D según longitud total del enlace
        # Objetivo: mantener ~400 puntos para fluidez óptima
        if len(consZ_np) > 1:
            total_length_km = (consZ_np[-1] - consZ_np[0]) / 1000.0  # Longitud total en km
            z_diff_m = abs(consZ_np[1] - consZ_np[0]) if len(consZ_np) > 1 else step_const_km * 1000
            z_diff_km = z_diff_m / 1000.0  # Paso de captura real
            
            # Calcular paso adaptativo para tener ~400 puntos máximo
            target_points = 400  # Número óptimo de puntos para visualización fluida
            total_available_points = len(consZ_np)
            
            if total_available_points <= target_points:
                # Si tenemos menos de 400 puntos, usamos todos
                const3d_every_calculated = 1
            else:
                # Si tenemos más, calculamos el paso para llegar a ~400
                const3d_every_calculated = max(1, int(round(total_available_points / target_points)))
            
            # Log para debug
            actual_3d_points = total_available_points // const3d_every_calculated
            print(f"[3D Adaptativo] Enlace: {total_length_km:.1f} km | "
                  f"Capturados: {total_available_points} pts | "
                  f"Every: {const3d_every_calculated} | "
                  f"Visualizados: {actual_3d_points} pts")
        else:
            const3d_every_calculated = 1

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
                every=const3d_every_calculated, pts_per_slice=const3d_html_pts, marker_size=2.0,
                trace_symbols=trace_symbols, num_traces=num_traces,
                group_by_quadrant=group_by_quadrant, show_slice_planes=show_slice_planes,
                chain=chain
            )

    if do_eye:
        save_eyediagram(Aout_np, sps, delay_total, plots_p / "eye.png")
    
    # Guardar waveforms TX vs RX si está habilitado
    if do_waveform:
        try:
            from .core import waveform as wf_module
            
            # Calcular segmento dinámico para VISUALIZACIÓN: mostrar exactamente 50 símbolos
            # Esto se adapta automáticamente a cualquier tasa de símbolos
            symbols_to_show = 50
            Rb = parGlob['Rb']
            Rs = Rb / math.log2(M_tx)  # Symbol rate
            T_symbol = 1.0 / Rs  # Duración de un símbolo en segundos
            segment_duration_us = symbols_to_show * T_symbol * 1e6  # Convertir a microsegundos
            
            # Calcular cantidad de muestras para el segmento de visualización
            segment_length_samples_plot = int(symbols_to_show * sps)
            
            # Guardar HDF5 con metadata
            wf_metadata = {
                'Fs': parGlob['Fs'],
                'sps': sps,
                'Ptx': parGlob['Ptx'],
                'Nsym': parGlob['Nsym'],
                'Rb': parGlob['Rb'],
                'Rs': Rs,
                'mod': mod,
                'rx': rx_mode,
                'backend': backend_info,
                'symbols_displayed': symbols_to_show
            }
            
            # Aplicar matched filter RRC para obtener señal post-DSP
            # Esto muestra el efecto del filtrado en la señal recibida
            Aout_post_mf = None
            try:
                rolloff = float(pulse_par.get("rolloff", 0.5))
                span_rrc = int(pulse_par.get("span", 8))
                Aout_post_mf = dsp.rrc_matched_filter_np(Aout_np, sps, rolloff, span_rrc)
            except Exception as e_mf:
                rprint(f"[yellow]Advertencia: No se pudo aplicar matched filter para waveform: {e_mf}[/yellow]")
            
            # Guardar TODAS las muestras en HDF5 (no solo las de visualización)
            wf_module.save_waveforms_hdf5(
                tx_signal=_to_numpy_if_needed(Ein, xp),
                rx_signal=Aout_np,
                rx_signal_post_mf=Aout_post_mf,
                filepath=plots_p / "waveforms.h5",
                metadata=wf_metadata,
                segment_start=0,
                segment_length=None  # None = guardar todo
            )
            
            # Calcular SNR medido en cada etapa (para debugging)
            def calc_snr_vs_clean(noisy, clean):
                """Calcula SNR comparando señal ruidosa vs limpia"""
                n = min(len(noisy), len(clean))
                if n == 0:
                    return None
                sig_power = np.mean(np.abs(clean[:n])**2)
                noise = noisy[:n] - clean[:n]
                noise_power = np.mean(np.abs(noise)**2)
                if noise_power <= 0:
                    return None
                return 10 * np.log10(sig_power / noise_power)
            
            # Señal TX limpia (sin AWGN)
            Ein_clean = _to_numpy_if_needed(xp.sqrt(parGlob["Ptx"]) * txSig, xp)
            Ein_actual = _to_numpy_if_needed(Ein, xp)
            
            snr_tx_measured = calc_snr_vs_clean(Ein_actual, Ein_clean)
            if snr_tx_measured is not None:
                print(f"[SNR Medido TX]: {snr_tx_measured:.2f} dB (objetivo: {parGlob.get('awgn_tx_snr_db', 'N/A')} dB)")
            
            # Crear gráfico comparativo con 3 paneles: TX, RX pre-DSP, RX post-MF
            wf_module.plot_waveform_comparison(
                tx_signal=_to_numpy_if_needed(Ein, xp),
                rx_signal=Aout_np,  # Pre-DSP (después de propagación óptica)
                sps=sps,
                Fs=parGlob['Fs'],
                segment_start_us=0.0,
                segment_length_us=segment_duration_us,
                filepath=plots_p / "waveform_comparison.png",
                rx_post_dsp=Aout_post_mf  # Post-DSP (después de matched filter)
            )
            rprint(f"[green]Waveforms guardados: {symbols_to_show} símbolos ({segment_duration_us:.2f} μs)[/green]")
        except Exception as e:
            rprint(f"[yellow]Advertencia: No se pudieron guardar waveforms: {e}[/yellow]")

    rprint(f"[bold cyan]{backend_info}[/bold cyan]")
    rprint(f"[bold green]Listo[/bold green]: log en [cyan]{outdir}/{log_name}[/cyan], "
           f"plots en [cyan]{plots_dir}[/cyan].")
    ber_print = result.get("BER_est_BPSK")
    if ber_print is None:
        ber_print = result.get("BER_post")
    ber_str = (f"{ber_print:.3e}" if isinstance(ber_print, (int, float)) and ber_print is not None else "n/a")
    
    # Mensaje simplificado: solo BER y OSNR (métricas ópticas relevantes)
    msg = f"L = {result['Lcum_m']/1e3:.1f} km | G = {result['G_dB']:.1f} dB | elapsed = {elapsed:.3f} s | BER={ber_str}"
    if osnr_final_db is not None:
        msg += f" | OSNR={osnr_final_db:.2f} dB"
    rprint(msg)

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
    step_const_km: float = typer.Option(0.5, help="Paso de captura de datos (fijo en 0.5 km)."),
    step_plot2d_km: float = typer.Option(5.0, help="Paso para graficar constelaciones 2D en km."),
    do_eye: bool = typer.Option(True, help="Guardar eye diagram al final."),
    plots_dir: str = typer.Option("plots", help="Carpeta para imágenes."),
    do_const3d: bool = typer.Option(False, help="Guardar PNG 3D con matplotlib."),
    const3d_every: int = typer.Option(1, help="Usar 1 de cada N snapshots en 3D PNG."),
    const3d_pts: int = typer.Option(1000, help="Máx. puntos por snapshot para PNG 3D."),
    do_const3d_html: bool = typer.Option(True, help="Guardar 3D interactivo HTML con Plotly."),
    step_const3d_km: float = typer.Option(0.5, help="Paso para visualización 3D (fijo en 0.5 km)."),
    const3d_html_pts: int = typer.Option(1200, help="Máx. puntos por snapshot para HTML 3D."),
    trace_symbols: bool = typer.Option(True, help="Modo trayectorias (seguir símbolos individuales)."),
    num_traces: int = typer.Option(50, help="Número de símbolos a seguir en modo trayectorias."),
    group_by_quadrant: bool = typer.Option(True, help="Agrupar símbolos por cuadrante QPSK (mismo color)."),
    show_slice_planes: bool = typer.Option(True, help="Mostrar planos semitransparentes en posiciones clave."),
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
        step_plot2d_km=step_plot2d_km,
        do_eye=do_eye,
        plots_dir=plots_dir,
        do_const3d=do_const3d,
        const3d_every=const3d_every,
        const3d_pts=const3d_pts,
        do_const3d_html=do_const3d_html,
        step_const3d_km=step_const3d_km,
        const3d_html_pts=const3d_html_pts,
        trace_symbols=trace_symbols,
        num_traces=num_traces,
        group_by_quadrant=group_by_quadrant,
        show_slice_planes=show_slice_planes,
    )

if __name__ == "__main__":
    app()
