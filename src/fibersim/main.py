from __future__ import annotations
import os, json, time, pathlib, importlib, math
from typing import Any, Dict
import typer
from rich import print as rprint
import numpy as np

app = typer.Typer(help="Simulador de fibra con RRC + SSFM + plots 2D y 3D.")

# ------------------------- util -------------------------

def calculate_system_noise_snr(ptx_dbm: float) -> tuple[float, float]:
    """Calcula SNR de ruido del sistema basado en potencia TX.
    
    Modelo físico extremo:
    - A bajas Ptx: dominado FUERTEMENTE por ruido térmico del receptor (constante)
    - A Ptx óptima (0 a +2 dBm): mínimo ruido, máximo SNR
    - A altas Ptx: aumenta shot noise
    
    Curva calibrada:
    Ptx = -20 dBm → SNR ≈ 0-1 dB (casi inutilizable, puro ruido)
    Ptx = -10 dBm → SNR ≈ 1-3 dB (muy degradado, constelación muy difusa)
    Ptx = -6 dBm → SNR ≈ 4-12 dB (degradado visible)
    Ptx = -3 dBm → SNR ≈ 13-18 dB (empieza a mejorar)
    Ptx = 0 dBm → SNR ≈ 28-30 dB (óptimo)
    Ptx = +2 dBm → SNR ≈ 30-32 dB (excelente - sweet spot)
    Ptx = +5 dBm → SNR ≈ 25 dB (baja por shot noise)
    
    Args:
        ptx_dbm: Potencia de transmisión en dBm
        
    Returns:
        (snr_tx_db, snr_rx_db): SNR para TX y RX en dB
    """
    # Punto óptimo y SNR máximo
    P_tx_optimal_dbm = 1.0  # Óptimo en +1 dBm
    snr_max = 32.0  # SNR máximo alcanzable
    
    # Modelo extremo de ruido térmico
    # A bajas potencias, el SNR crece muy lentamente con Ptx
    # Factor de escala para hacer la curva más extrema
    
    if ptx_dbm <= -15:
        # Zona de ruido térmico dominante total: SNR muy bajo
        snr_rx_db = 0.5 + (ptx_dbm + 20) * 0.1  # Crece muy lento
        snr_rx_db = max(0.3, min(2.0, snr_rx_db))
        
    elif ptx_dbm <= -10:
        # Transición -15 a -10: SNR de 1.5 a 3 dB
        t = (ptx_dbm + 15) / 5.0  # 0 a 1
        snr_rx_db = 1.5 + t * 1.5
        
    elif ptx_dbm <= -6:
        # Transición -10 a -6: SNR de 3 a 10 dB (subida moderada)
        t = (ptx_dbm + 10) / 4.0  # 0 a 1
        snr_rx_db = 3.0 + t * 7.0
        
    elif ptx_dbm <= -3:
        # Transición -6 a -3: SNR de 10 a 17 dB (mejora más rápida)
        t = (ptx_dbm + 6) / 3.0  # 0 a 1
        snr_rx_db = 10.0 + t * 7.0
        
    elif ptx_dbm <= 0:
        # Transición -3 a 0: SNR de 17 a 28 dB (rápida mejora)
        t = (ptx_dbm + 3) / 3.0  # 0 a 1
        # Curva acelerada
        snr_rx_db = 17.0 + (t ** 0.7) * 11.0
        
    elif ptx_dbm <= 2:
        # Zona óptima 0 a +2 dBm: SNR máximo 28-32 dB
        t = ptx_dbm / 2.0  # 0 a 1
        snr_rx_db = 28.0 + t * 4.0
        
    else:
        # Ptx > +2 dBm: empieza a bajar por shot noise
        delta = ptx_dbm - 2.0
        snr_rx_db = snr_max - (delta * 0.7)  # Baja ~0.7 dB por cada dB extra
        snr_rx_db = max(15.0, snr_rx_db)  # Mínimo 15 dB a altas potencias
    
    # Garantizar límites físicos
    snr_rx_db = max(0.3, min(snr_max, snr_rx_db))
    
    # SNR del transmisor: mejor que RX
    # A bajas potencias, TX también tiene ruido pero menos crítico
    if ptx_dbm <= -10:
        snr_tx_db = snr_rx_db + 3.0  # Diferencia pequeña a bajas Ptx
    else:
        snr_tx_db = snr_rx_db + 6.0  # Diferencia mayor a Ptx normales
    
    return snr_tx_db, snr_rx_db

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
    
    # Session ID para multiusuario (evitar sobrescritura de archivos)
    session_id = parGlob.get("session_id", "default")
    
    # Paths
    outdir_p = pathlib.Path(outdir)
    outdir_p.mkdir(exist_ok=True)
    plots_p = pathlib.Path(parGlob.get("plots_dir", "plots"))
    plots_p.mkdir(exist_ok=True)

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
        # Calcular SNR automáticamente basado en Ptx
        ptx_dbm = 10.0 * math.log10(parGlob["Ptx"] * 1000.0)  # W -> dBm
        snr_tx, snr_rx_target = calculate_system_noise_snr(ptx_dbm)
        
        print(f"[AWGN] Ptx={ptx_dbm:.2f} dBm → SNR_TX={snr_tx:.1f} dB, SNR_RX={snr_rx_target:.1f} dB")
        print(f"[AWGN] Aplicando ruido TX: SNR={snr_tx:.1f} dB")
        Ein, P_awgn_tx = awgn.add_awgn(Ein, snr_tx, xp, sps=1, rolloff=0.0, mode="sample")
        info["P_AWGN_TX"] = P_awgn_tx
        info["SNR_RX_TARGET"] = snr_rx_target  # Guardar para usar en RX
    else:
        print(f"[AWGN] DESACTIVADO")
        info["P_AWGN_TX"] = 0.0
        info["SNR_RX_TARGET"] = None

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
        # Usar SNR calculado previamente basado en Ptx
        snr_rx = info.get("SNR_RX_TARGET", 25.0)
        print(f"[AWGN] Aplicando ruido RX: SNR={snr_rx:.1f} dB")
        Aout, P_awgn_rx = awgn.add_awgn(Aout, snr_rx, xp, sps=1, rolloff=0.0, mode="sample")
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

    # Nombre del log con timestamp
    log_name = f"simlog_{time.strftime('%Y-%m-%d_%H-%M-%S')}.json"

    # ---------------- RX / métricas ----------------
    sps = int(diag.get("sps", parGlob["sps"]))
    span = int(pulse_par.get("span", 8))
    # delay_samp in diag already accounts for TX+RX filter group delay (≈ span*sps)
    delay_guess = int(diag.get("delay_samp", 0))

    Aout_np = _to_numpy_if_needed(Aout, xp)
    syms_np = _to_numpy_if_needed(syms, xp)
    Nsym = int(parGlob["Nsym"])
    
    # ============ CÁLCULO DE SNR PRE-DSP (estimado teóricamente) ============
    # El matched filter proporciona una ganancia de procesamiento aproximada de 10*log10(sps) dB
    # Por lo tanto: SNR_pre_DSP ≈ SNR_post_DSP - ganancia_MF
    # Este es el approach estándar en teoría de comunicaciones
    snr_pre_dsp_db = None
    
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

    # ============ DEMODULACIÓN MEJORADA ============
    # El matched filter RX ya fue aplicado en run_chain, trabajamos con Aout (señal filtrada)
    # NO necesitamos volver a aplicar matched filter aquí
    
    # Para BPSK IMDD: usar método de búsqueda de delay óptimo
    if (M_tx == 2) and (str(rx_mode).lower() == "imdd"):
        print(f"[Demod] Modo BPSK IMDD - búsqueda de delay óptimo")
        best_delay, BER_est, s_hat = modem.find_best_delay(
            rx_wave=Aout_np,
            sps=sps,
            tx_syms_ref=syms_np[:Nsym],
            guess_delay=delay_guess,
            halfwin=max(8, sps * 2),
        )
        delay_total = int(best_delay)
        snr_sym_db = None  # No calculamos SNR para IMDD
    else:
        # Para QPSK y 16-QAM: demodulación coherente estándar
        print(f"[Demod] Modo {mod} coherente - demodulación estándar")
        
        # Extracción de símbolos con delay conocido
        s_hat = modem.slice_to_symbols(Aout_np, sps=sps, delay_samp=delay_guess, Nsym=Nsym)
        delay_total = int(delay_guess)
        
        # TX reference: slice_to_symbols ya maneja el delay en muestras,
        # así que los símbolos RX corresponden a los primeros símbolos TX
        tx_ref = syms_np[: len(s_hat)]
        
        # Alineación de fase para QPSK/16-QAM (MMSE)
        if M_tx > 2:
            s_hat = modem.carrier_phase_align(s_hat, tx_ref)
        
        # Cálculo de BER (sin normalizar - importante para precisión)
        BER_est = modem.ber_from_symbols(tx_ref, s_hat, M=M_tx)
        
        # Cálculo de SNR post-DSP usando método mejorado
        try:
            from .core.snr import calc_snr_post_dsp
            _, snr_sym_db = calc_snr_post_dsp(s_hat, tx_ref)
            
            # Calcular SNR pre-DSP estimado (antes del matched filter RX)
            # La ganancia de procesamiento del matched filter es aproximadamente 10*log10(sps)
            # SNR_pre = SNR_post - ganancia_procesamiento
            matched_filter_gain_db = 10.0 * math.log10(sps)
            snr_pre_dsp_db = snr_sym_db - matched_filter_gain_db
            print(f"[SNR] Post-DSP: {snr_sym_db:.2f} dB | Pre-DSP (estimado): {snr_pre_dsp_db:.2f} dB | Ganancia MF: {matched_filter_gain_db:.2f} dB")
        except Exception as e:
            print(f"[Demod] Advertencia: No se pudo calcular SNR post-DSP: {e}")
            snr_sym_db = None
    
    print(f"[Demod] BER: {BER_est:.6e} | SNR: {snr_sym_db:.2f} dB" if snr_sym_db else f"[Demod] BER: {BER_est:.6e}")
    # ===============================================

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
        "notes": "RRC+SSFM con demodulación mejorada; snapshots guardados si do_const=True",
        "Lcum_m": info.get("Lcum", 0.0),
        "G_dB": info.get("G_dB", 0.0),
        "Pmean_W": info.get("Pmean", None),
        "backend": backend_info,
        "elapsed_s": elapsed,
        "delay_best_samp": delay_total,
        "BER": BER_est,  # BER unificado para todas las modulaciones
        "SNR_sym_dB": snr_sym_db,  # SNR medido post-DSP (después del matched filter)
        "OSNR_final_dB": osnr_final_db,  # OSNR óptico (solo ASE)
        "SNR_pre_dsp_dB": snr_pre_dsp_db,  # SNR pre-DSP (antes del matched filter, medido en waveform)
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

        save_constellations_grid(consSym_plot2d, consZ_plot2d, plots_p / f"constelaciones_{session_id}.png")
        save_power_evolution(powZ_np, powW_np, plots_p / f"potencia_{session_id}.png", unit="dBm")

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
                plots_p / f"constelaciones_3d_{session_id}.png",
                every=const3d_every, pts_per_slice=const3d_pts, marker_size=1.0
            )
        if do_const3d_html:
            save_constellations_3d_html(
                consSym_np, consZ_np,
                plots_p / f"constelaciones_3d_{session_id}.html",
                every=const3d_every_calculated, pts_per_slice=const3d_html_pts, marker_size=2.0,
                trace_symbols=trace_symbols, num_traces=num_traces,
                group_by_quadrant=group_by_quadrant, show_slice_planes=show_slice_planes,
                chain=chain
            )

    if do_eye:
        # Usar señal después del matched filter (más limpia, sin ruido AWGN visible)
        # Aout_mf es la señal ya filtrada disponible en diag
        signal_for_eye = diag.get("A_rx_matched", Aout_np) if "A_rx_matched" in diag else Aout_np
        save_eyediagram(signal_for_eye, sps, delay_total, plots_p / f"eye_{session_id}.png")
    
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
                filepath=plots_p / f"waveforms_{session_id}.h5",
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
                filepath=plots_p / f"waveform_comparison_{session_id}.png",
                rx_post_dsp=Aout_post_mf  # Post-DSP (después de matched filter)
            )
            rprint(f"[green]Waveforms guardados: {symbols_to_show} símbolos ({segment_duration_us:.2f} μs)[/green]")
        except Exception as e:
            rprint(f"[yellow]Advertencia: No se pudieron guardar waveforms: {e}[/yellow]")

    rprint(f"[bold cyan]{backend_info}[/bold cyan]")
    rprint(f"[bold green]Listo[/bold green]: log en [cyan]{outdir}/{log_name}[/cyan], "
           f"plots en [cyan]{plots_dir}[/cyan].")
    
    # Obtener BER del campo correcto
    ber_print = result.get("BER")
    ber_str = (f"{ber_print:.3e}" if isinstance(ber_print, (int, float)) and ber_print is not None else "n/a")
    
    # Mensaje completo con todas las métricas importantes
    msg = f"[Session: {session_id}] L = {result['Lcum_m']/1e3:.1f} km | G = {result['G_dB']:.1f} dB | elapsed = {elapsed:.3f} s"
    msg += f" | BER={ber_str}"
    
    # Agregar SNR pre y post DSP si están disponibles
    snr_pre = result.get("SNR_pre_dsp_dB")
    snr_post = result.get("SNR_sym_dB")
    if snr_pre is not None and snr_post is not None:
        msg += f" | SNR: {snr_pre:.2f} dB (pre) → {snr_post:.2f} dB (post)"
    elif snr_post is not None:
        msg += f" | SNR post-DSP: {snr_post:.2f} dB"
    
    # Agregar OSNR si está disponible
    if osnr_final_db is not None:
        msg += f" | OSNR: {osnr_final_db:.2f} dB"
    
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
