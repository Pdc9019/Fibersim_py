# src/fibersim/gui/app.py
from __future__ import annotations
import json, pathlib, uuid, math, io, csv, copy, datetime
from typing import Any, Dict, List
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import numpy as np

# Modelos / ejecución del simulador
from fibersim.schema import SimConfig, FiberBlock, EdfaBlock, FiberPar, EdfaPar
from fibersim.main import _execute

# ------------------------- Page Configuration -------------------------
st.set_page_config(
    page_title="FiberSim - Simulador de Enlaces de Fibra Óptica",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Main content area */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-weight: 600;
        letter-spacing: -0.02em;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 600;
    }
    
    /* Remove extra spacing */
    .element-container {
        margin-bottom: 0.5rem;
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 0.5rem;
    }
    
    /* Buttons */
    .stButton>button {
        border-radius: 0.375rem;
        font-weight: 500;
        transition: all 0.2s;
    }
    
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Sidebar improvements */
    [data-testid="stSidebar"] {
        background-color: rgba(38, 39, 48, 0.05);
    }
    
    [data-testid="stSidebar"] .element-container {
        margin-bottom: 0.75rem;
    }
    
    /* Sidebar title styling */
    [data-testid="stSidebar"] h1 {
        font-size: 1.5rem;
        margin-bottom: 1rem;
    }
    
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        font-size: 1.1rem;
        margin-top: 0.5rem;
        margin-bottom: 0.75rem;
    }
    
    /* Hacer selectbox más visible en sidebar */
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(250, 250, 250, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# ------------------------- utilidades -------------------------

def ensure_uid(blk: dict) -> None:
    if "_uid" not in blk:
        blk["_uid"] = uuid.uuid4().hex

def summarize_block(blk: dict) -> str:
    t = blk["type"]
    if t == "fiber":
        p = blk["par"]
        L_km = p["L"] / 1e3
        alpha_db_km = 4.343 * p["alpha"] * 1e3
        return f"{L_km:.0f} km · α≈{alpha_db_km:.2f} dB/km"
    elif t == "edfa":
        p = blk["par"]
        return f"G {p['G_dB']:.1f} dB · nsp {p['nsp']:.2f}"
    return t.upper()

def move_block(idx: int, delta: int):
    j = idx + delta
    if 0 <= j < len(st.session_state.chain):
        st.session_state.chain[idx], st.session_state.chain[j] = (
            st.session_state.chain[j],
            st.session_state.chain[idx],
        )

def move_block_to(idx_from: int, idx_to_1based: int):
    n = len(st.session_state.chain)
    if n == 0: return
    idx_to = max(0, min(n - 1, int(idx_to_1based) - 1))
    if idx_to == idx_from: return
    blk = st.session_state.chain.pop(idx_from)
    st.session_state.chain.insert(idx_to, blk)

def delete_block(idx: int):
    st.session_state.chain.pop(idx)

def duplicate_block(idx: int):
    import copy
    blk = copy.deepcopy(st.session_state.chain[idx])
    blk["_uid"] = uuid.uuid4().hex
    st.session_state.chain.insert(idx + 1, blk)

def w_to_dbm(p_w: float) -> float:
    return 10.0 * math.log10(max(p_w, 1e-30) / 1e-3)

def dbm_to_w(dbm: float) -> float:
    return 1e-3 * (10.0 ** (dbm / 10.0))

# ------------------------- descubrimiento de ejemplos -------------------------

# Descubrir archivos JSON en examples/configs que validen contra el esquema
def discover_example_files() -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    try:
        base = pathlib.Path(__file__).resolve().parents[3]  # repo root
        exdir = base / "examples" / "configs"
        if not exdir.exists():
            return out
        for p in sorted(exdir.glob("*.json")):
            try:
                raw = json.loads(p.read_text(encoding="utf-8"))
                cfg = SimConfig.model_validate(raw)
                out[p.name] = cfg.model_dump(by_alias=True)
            except Exception:
                # ignorar archivos que no cumplan el esquema
                continue
    except Exception:
        pass
    return out

# ------------------------- página / estado -------------------------

st.set_page_config(page_title="FiberSim GUI", layout="wide")

st.markdown("""
<style>
div.block-container { padding-top: 1.0rem; }
.mini-card { border: 1px solid rgba(255,255,255,.12); background: rgba(255,255,255,.03);
  border-radius: 10px; padding: 8px 10px; margin: 6px 0; }
.mini-card:hover { background: rgba(255,255,255,.05); }
.mini-head { display:flex; align-items:center; gap:.45rem; white-space:nowrap; overflow:hidden; }
.mini-idx { font-weight:700; opacity:.85; }
.mini-sub { opacity:.82; font-size:.85rem; overflow:hidden; text-overflow:ellipsis; }
.chip { font-size:.78rem; padding:.16rem .45rem; border-radius:8px; font-weight:700; }
.chip.fiber { background:#1e3a8a22; color:#93c5fd; border:1px solid #1e3a8a44; }
.chip.edfa  { background:#14532d22; color:#86efac; border:1px solid #14532d44; }
div[data-testid="stHorizontalBlock"] { gap:.25rem; }

/* Chips de meta (backend/SNR/OSNR/BER) */
.meta-row {font-size:0.90rem; opacity:.95; display:flex; gap:.6rem; flex-wrap:wrap; margin:.35rem 0 0.4rem 0;}
.meta-chip{padding:.15rem .5rem; border:1px solid rgba(255,255,255,.25); border-radius:8px; background:rgba(255,255,255,.04)}
</style>
""", unsafe_allow_html=True)

# Estado inicial
# Generar session_id único para multiusuario
if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())[:8]
    # Log en consola cuando se crea nueva sesión
    print(f"\n{'='*60}")
    print(f"🟢 NUEVA SESIÓN CONECTADA: {st.session_state.session_id}")
    print(f"{'='*60}\n")

if "chain" not in st.session_state:
    st.session_state.chain = []
if "global" not in st.session_state:
    st.session_state["global"] = dict(Rb=10e9, M=2, sps=8, Fs=80e9, Nsym=32768, Ptx=1e-3, lambda_nm=1550.0, mod="BPSK", rx="imdd", pol="sp")
if "pulse" not in st.session_state:
    st.session_state["pulse"] = dict(type="RRC", roll=0.2, span=8)
if "edit_idx" not in st.session_state:
    st.session_state.edit_idx = None
if "last_backend" not in st.session_state:
    st.session_state["last_backend"] = None

# Función de limpieza de archivos antiguos (>30 min)
def cleanup_old_plots(plots_dir: Path, max_age_minutes: int = 30):
    """Elimina archivos de plots más antiguos que max_age_minutes."""
    import time
    try:
        current_time = time.time()
        for file_path in plots_dir.glob("*"):
            if file_path.is_file():
                file_age_minutes = (current_time - file_path.stat().st_mtime) / 60
                if file_age_minutes > max_age_minutes:
                    try:
                        file_path.unlink()
                    except Exception:
                        pass  # Ignorar si no se puede eliminar
    except Exception:
        pass  # Ignorar errores de limpieza

# Contar sesiones activas (archivos únicos en plots)
def count_active_sessions():
    """Cuenta sesiones activas basándose en archivos únicos con session_id."""
    try:
        from pathlib import Path
        import re
        plots_dir = Path("plots")
        if not plots_dir.exists():
            return 0, set()
        
        # Buscar todos los archivos con patrón *_sessionid.png
        session_ids = set()
        # Patrón para session_id válido: 8 caracteres hexadecimales (UUID corto)
        uuid_pattern = re.compile(r'^[a-f0-9]{8}$')
        
        for file in plots_dir.glob("*_*.png"):
            # Extraer session_id del nombre (formato: tipo_sessionid.png)
            parts = file.stem.split("_")
            if len(parts) >= 2:
                potential_id = parts[-1]  # Último elemento
                # Validar que sea un UUID de 8 chars hex
                if uuid_pattern.match(potential_id):
                    session_ids.add(potential_id)
        
        return len(session_ids), session_ids
    except Exception:
        return 0, set()

# Mostrar sesiones activas en consola
active_count, active_ids = count_active_sessions()
if active_count > 0:
    print(f"\n📊 SESIONES ACTIVAS: {active_count}")
    print(f"   IDs: {', '.join(sorted(active_ids))}")
    print()

# Ejecutar limpieza al iniciar (solo una vez por sesión)
if "cleanup_done" not in st.session_state:
    from pathlib import Path
    cleanup_old_plots(Path("plots"), max_age_minutes=30)
    st.session_state.cleanup_done = True
    print(f"🧹 Limpieza automática ejecutada (archivos >30 min eliminados)\n")

# ------------------------- carga / guardado y presets -------------------------

# Título principal del sidebar
st.sidebar.title("Gestión de Configuración")

# Mostrar sesiones activas
active_sessions, _ = count_active_sessions()
st.sidebar.caption(f"🔑 ID de sesión: `{st.session_state.session_id}`")
st.sidebar.markdown("---")

# Sección 1: Cargar configuración personalizada (expuesta directamente)
st.sidebar.markdown("**Cargar Configuración**")
upl = st.sidebar.file_uploader("Seleccionar archivo JSON", type=["json"], label_visibility="collapsed")
if upl:
    try:
        cfg = SimConfig.model_validate(json.loads(upl.read().decode("utf-8")))
        st.session_state["global"] = cfg.global_.model_dump()
        st.session_state["pulse"]  = cfg.pulse.model_dump()
        st.session_state.chain     = [b.model_dump() for b in cfg.chain]
        for b in st.session_state.chain: ensure_uid(b)
        st.session_state.edit_idx = None
        st.sidebar.success("Configuración cargada")
    except Exception as e:
        st.sidebar.error(f"Error al cargar: {e}")

st.sidebar.markdown("")

# Sección 2: Guardar configuración actual
def export_json() -> str:
    data = {"global": st.session_state["global"], "pulse": st.session_state["pulse"], "chain": st.session_state.chain}
    return json.dumps(data, indent=2)

def generate_unique_filename() -> str:
    """Genera un nombre de archivo único y descriptivo basado en la configuración actual."""
    # Extraer parámetros clave
    gp = st.session_state["global"]
    Rb_Gbps = float(gp.get("bit_rate", 1e9)) / 1e9
    M_order = int(gp.get("M", 4))
    
    # Mapeo de modulación
    mod_names = {2: "BPSK", 4: "QPSK", 16: "16QAM", 64: "64QAM"}
    mod_name = mod_names.get(M_order, f"{M_order}QAM")
    
    # Calcular longitud total de fibra
    total_length_km = sum(blk["par"]["L"] / 1000.0 for blk in st.session_state.chain if blk["type"] == "fiber")
    
    # Timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Formato: fibersim_MODULACION_XXGbps_YYYkm_TIMESTAMP.json
    filename = f"fibersim_{mod_name}_{Rb_Gbps:.0f}Gbps_{total_length_km:.0f}km_{timestamp}.json"
    
    return filename

st.sidebar.download_button(
    "💾 Guardar Configuración Actual", 
    data=export_json(), 
    file_name=generate_unique_filename(), 
    mime="application/json",
    use_container_width=True,
    help="Descarga la configuración actual como archivo JSON con nombre descriptivo automático"
)

st.sidebar.markdown("---")

# Sección 3: Enlaces de ejemplo
st.sidebar.subheader("Enlaces de Ejemplo")

files_map = discover_example_files()
if files_map:
    file_opts = sorted(files_map.keys())
    
    # Selector con mejor presentación
    file_sel = st.sidebar.selectbox(
        "Seleccionar configuración de ejemplo",
        options=file_opts,
        index=0,
        help="Ejemplos validados y listos para simular"
    )
    
    # Botón para cargar
    if st.sidebar.button("Cargar Ejemplo", use_container_width=True, type="primary"):
        try:
            raw = files_map[file_sel]
            cfg = SimConfig.model_validate(raw)
            st.session_state["global"] = cfg.global_.model_dump()
            st.session_state["pulse"]  = cfg.pulse.model_dump()
            st.session_state.chain     = [b.model_dump() for b in cfg.chain]
            for b in st.session_state.chain: ensure_uid(b)
            st.session_state.edit_idx = None
            st.sidebar.success(f"'{file_sel}' cargado exitosamente")
        except Exception as e:
            st.sidebar.error(f"Error al cargar: {e}")
else:
    st.sidebar.info("No se encontraron archivos de ejemplo en `examples/configs/`")

# ------------------------- parámetros simples y robustos -------------------------

st.title("FiberSim | Simulador de Enlaces de Fibra Óptica")
st.markdown("---")

gcol, pcol = st.columns(2)

with gcol:
    st.markdown("### Parámetros Globales")

    g = st.session_state["global"]

    # Entrada en Gbaud para evitar números enormes y minimizar errores
    col1, col2 = st.columns(2)
    with col1:
        Rb_gbaud = st.number_input(
            "Tasa de Símbolos", 
            value=float(g.get("Rb", 10e9)) / 1e9, 
            min_value=0.1, 
            step=0.1, 
            format="%.2f",
            help="Tasa de símbolos en Gbaud (GSymbolos/s)"
        )
    with col2:
        st.metric("Gbaud", f"{Rb_gbaud:.2f}")
    
    # Potencia en dBm, convertimos internamente a W
    pt_dbm_default = w_to_dbm(float(g["Ptx"]))
    col1, col2 = st.columns(2)
    with col1:
        Ptx_dBm = st.number_input(
            "Potencia TX", 
            value=float(pt_dbm_default), 
            step=0.1, 
            format="%.2f",
            help="Potencia óptica de transmisión en dBm"
        )
    with col2:
        st.metric("dBm", f"{Ptx_dBm:.2f}")

    # Calidad controla Nsym
    calidad = st.selectbox(
        "Calidad de Simulación", 
        ["Rápida", "Media", "Alta"], 
        index=1,
        help="Controla el número de símbolos y precisión de BER: Rápida=16384, Media=32768, Alta=65536"
    )
    
    # Modulación
    mod_map_display = {2: "2 - BPSK", 4: "4 - QPSK", 16: "16 - 16QAM"}
    mod_map_value = {"2 - BPSK": 2, "4 - QPSK": 4, "16 - 16QAM": 16}
    
    M_now = int(g.get("M", 2))
    M_display_now = mod_map_display[M_now]
    
    M_display_sel = st.selectbox(
        "Orden de Modulación", 
        ["2 - BPSK", "4 - QPSK", "16 - 16QAM"],
        index=["2 - BPSK", "4 - QPSK", "16 - 16QAM"].index(M_display_now),
        help="Esquema de modulación digital: BPSK (1 bit/símbolo), QPSK (2 bits/símbolo), 16-QAM (4 bits/símbolo)"
    )
    
    M_sel = mod_map_value[M_display_sel]
    
    # Receptor (solo si BPSK)
    if M_sel == 2:
        rx_sel = st.selectbox(
            "Tipo de Receptor", 
            ["imdd", "coh"], 
            index=["imdd", "coh"].index(str(g.get("rx", "imdd"))),
            help="IMDD: Detección directa (solo intensidad) | COH: Coherente (preserva fase)"
        )
    else:
        rx_sel = "coh"
        st.caption("QPSK/16-QAM requieren receptor coherente (automático)")
    
    # Longitud de onda
    lambda_nm = st.number_input(
        "Longitud de Onda [nm]", 
        value=float(g.get("lambda_nm", 1550.0)),
        min_value=1200.0, 
        max_value=1650.0, 
        step=1.0,
        help="Longitud de onda óptica. Estándar banda C: 1530-1565 nm. Afecta cálculos de ASE y dispersión cromática."
    )

    # Aplicar valores
    g["sps"] = 8  # Fijo
    g["Nsym"] = {"Rápida": 16384, "Media": 32768, "Alta": 65536}[calidad]
    g["lambda_nm"] = float(lambda_nm)
    g["M"] = M_sel
    g["mod"] = {2: "BPSK", 4: "QPSK", 16: "16QAM"}[M_sel]
    g["rx"] = rx_sel

    g["Rb"] = float(Rb_gbaud) * 1e9
    
    # Fs = Rb × sps (oversampling intencional, NO es un bug)
    # Esto da más resolución al DSP y mejora la calidad de la simulación
    g["Fs"] = float(g["Rb"]) * int(g["sps"])
    
    # Calcular Rs solo para mostrar info
    Rs = float(g["Rb"]) / np.log2(int(g["M"]))  # Symbol rate
    
    g["Ptx"] = float(dbm_to_w(Ptx_dBm))

    st.caption(f"Tasa de Muestreo: {g['Fs']/1e9:.3f} GS/s (Rb={g['Rb']/1e9:.3f} Gbps × sps={g['sps']})  |  Potencia TX: {Ptx_dBm:.2f} dBm ({g['Ptx']*1000:.3f} mW)")

    # ========== Ruido AWGN ==========
    st.markdown("##### Ruido del Sistema")
    
    enable_awgn = st.checkbox(
        "Activar Ruido del Sistema",
        value=g.get("enable_awgn", False),
        help="""Activa el modelo de ruido del sistema basado en física realista.
        
**Modelo Automático SNR(Ptx)**:
- El ruido se calcula automáticamente según la potencia TX
- A **bajas potencias** (<-10 dBm): dominado por ruido térmico del receptor → SNR muy bajo
- En la **zona óptima** (0 a +2 dBm): mínimo ruido → SNR máximo (~28-32 dB)
- A **altas potencias** (>5 dBm): aumenta shot noise → SNR baja gradualmente

**Simula**:
- Transmisor: Ruido del láser, jitter del modulador, imperfecciones electrónicas
- Receptor: Shot noise del fotodetector, ruido térmico del TIA, ruido del ADC

**Referencia**: Basado en modelos de receiver sensitivity y thermal noise floor.
Ver: Agrawal - "Fiber Optic Communications" (2007)"""
    )
    
    # Mostrar gráfico minimalista de la curva SNR(Ptx) cuando se activa
    if enable_awgn:
        import matplotlib.pyplot as plt
        
        # Usar columna pequeña para limitar el ancho
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col1:
            # Generar curva SNR(Ptx)
            ptx_range = np.linspace(-20, 10, 150)
            snr_rx_curve = []
            
            for ptx_dbm in ptx_range:
                snr_max = 32.0
                if ptx_dbm <= -15:
                    snr_rx_db = 0.5 + (ptx_dbm + 20) * 0.1
                    snr_rx_db = max(0.3, min(2.0, snr_rx_db))
                elif ptx_dbm <= -10:
                    t = (ptx_dbm + 15) / 5.0
                    snr_rx_db = 1.5 + t * 1.5
                elif ptx_dbm <= -6:
                    t = (ptx_dbm + 10) / 4.0
                    snr_rx_db = 3.0 + t * 7.0
                elif ptx_dbm <= -3:
                    t = (ptx_dbm + 6) / 3.0
                    snr_rx_db = 10.0 + t * 7.0
                elif ptx_dbm <= 0:
                    t = (ptx_dbm + 3) / 3.0
                    snr_rx_db = 17.0 + (t ** 0.7) * 11.0
                elif ptx_dbm <= 2:
                    t = ptx_dbm / 2.0
                    snr_rx_db = 28.0 + t * 4.0
                else:
                    delta = ptx_dbm - 2.0
                    snr_rx_db = snr_max - (delta * 0.7)
                    snr_rx_db = max(15.0, snr_rx_db)
                snr_rx_curve.append(max(0.3, min(snr_max, snr_rx_db)))
            
            # Crear figura pequeña tipo badge
            fig, ax = plt.subplots(figsize=(1.2, 1.2))
            fig.patch.set_facecolor('#0E1117')  # Fondo oscuro de Streamlit
            ax.set_facecolor('#0E1117')
            
            # Curva principal
            ax.plot(ptx_range, snr_rx_curve, color='#4A9EFF', linewidth=1.5, alpha=0.9)
            ax.fill_between(ptx_range, 0, snr_rx_curve, color='#4A9EFF', alpha=0.08)
            
            # Zona óptima muy sutil
            ax.axvspan(-3, 2, alpha=0.05, color='#00FF00')
            
            # Etiquetas con estilo minimalista
            ax.set_xlabel('Ptx [dBm]', fontsize=6, color='#AAAAAA')
            ax.set_ylabel('SNR [dB]', fontsize=6, color='#AAAAAA')
            ax.grid(True, alpha=0.12, linestyle='-', linewidth=0.3, color='#444444')
            ax.set_xlim(-20, 10)
            ax.set_ylim(0, 35)
            
            # Tick labels pequeños y discretos
            ax.tick_params(labelsize=5, colors='#888888', length=1.5)
            
            # Marcadores de zona en texto muy pequeño
            ax.text(-15, 32, 'Tér', fontsize=5, color='#FF6B6B', alpha=0.5, ha='center')
            ax.text(0, 32, 'Ópt', fontsize=5, color='#51CF66', alpha=0.5, ha='center')
            ax.text(8, 32, 'Shot', fontsize=5, color='#FFA94D', alpha=0.5, ha='center')
            
            # Quitar bordes innecesarios
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#444444')
            ax.spines['bottom'].set_color('#444444')
            ax.spines['left'].set_linewidth(0.5)
            ax.spines['bottom'].set_linewidth(0.5)
            
            plt.tight_layout(pad=0.2)
            st.pyplot(fig)
            plt.close()
    
    g["enable_awgn"] = enable_awgn
    
   

def _generate_rrc_pulse(beta: float, span: int, sps: int = 8) -> tuple:
    """Genera pulso RRC para visualización en tiempo real"""
    N = span * sps
    t = np.arange(-N / 2, N / 2 + 1, dtype=np.float64) / sps
    taps = np.zeros_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-12:
            taps[i] = 1.0 + beta * (4 / np.pi - 1)
        elif abs(abs(4 * beta * ti) - 1.0) < 1e-12:
            taps[i] = (beta / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
            )
        else:
            num = np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))
            den = np.pi * ti * (1 - (4 * beta * ti) ** 2)
            taps[i] = num / den
    # Normalización de energía
    taps = taps / np.sqrt(np.sum(taps**2))
    return t, taps

with pcol:
    st.markdown("### Conformación de Pulso")
    
    # Sliders en la columna izquierda
    col_sliders, col_plot = st.columns([1, 1])
    
    with col_sliders:
        p = st.session_state["pulse"]
        p["type"] = "RRC"
        p["roll"] = st.slider(
            "Factor Roll-off (β)", 
            min_value=0.01, 
            max_value=1.0, 
            value=float(p.get("roll", 0.2)), 
            step=0.01,
            help="Parámetro de exceso de ancho de banda. BW = Rs × (1 + β)"
        )
        p["span"] = st.slider(
            "Ancho del Filtro (símbolos)", 
            min_value=4, 
            max_value=20, 
            value=int(p.get("span", 8)), 
            step=1,
            help="Duración del pulso en períodos de símbolo"
        )
    
    # Gráfico interactivo en la columna derecha
    with col_plot:
        # Generar pulso RRC con parámetros actuales
        t_pulse, h_pulse = _generate_rrc_pulse(
            beta=p["roll"], 
            span=p["span"], 
            sps=8
        )
        
        # Crear gráfico con Plotly
        fig_pulse = go.Figure()
        
        # Línea del pulso
        fig_pulse.add_trace(go.Scatter(
            x=t_pulse, 
            y=h_pulse,
            mode='lines',
            name='Pulso RRC',
            line=dict(color='#1f77b4', width=2.5),
            fill='tozeroy',
            fillcolor='rgba(31, 119, 180, 0.15)',
            hovertemplate='t = %{x:.2f} Ts<br>h(t) = %{y:.3f}<extra></extra>'
        ))
        
        # Marcar los instantes de símbolo (cruces por cero de ISI)
        symbol_times = np.arange(-p["span"]//2, p["span"]//2 + 1)
        symbol_values = np.interp(symbol_times, t_pulse, h_pulse)
        
        fig_pulse.add_trace(go.Scatter(
            x=symbol_times,
            y=symbol_values,
            mode='markers',
            name='Instantes símbolo',
            marker=dict(color='#d62728', size=7, symbol='cross', line=dict(width=1.5)),
            hovertemplate='Símbolo %{x}<br>ISI = %{y:.4f}<extra></extra>'
        ))
        
        # Configuración del layout
        fig_pulse.update_layout(
            title=dict(
                text=f"<b>Pulso RRC</b>  β={p['roll']:.2f}, span={p['span']}",
                font=dict(size=13),
                x=0.5,
                xanchor='center'
            ),
            xaxis_title="Tiempo (T<sub>símbolo</sub>)",
            yaxis_title="Amplitud",
            height=280,
            margin=dict(l=45, r=20, t=45, b=40),
            showlegend=False,
            hovermode='closest',
            template='plotly_white',
            plot_bgcolor='rgba(240,240,240,0.3)'
        )
        
        fig_pulse.update_xaxes(
            gridcolor='rgba(200,200,200,0.4)',
            zeroline=True,
            zerolinewidth=1.5,
            zerolinecolor='black',
            range=[-p["span"]/2 - 1, p["span"]/2 + 1]
        )
        
        fig_pulse.update_yaxes(
            gridcolor='rgba(200,200,200,0.4)',
            zeroline=True,
            zerolinewidth=1.5,
            zerolinecolor='black'
        )
        
        # Mostrar gráfico (key único para forzar actualización)
        st.plotly_chart(fig_pulse, use_container_width=True, key=f"pulse_viz_{p['roll']:.3f}_{p['span']}")

st.markdown("---")
# ------------------------- cadena (builder) -------------------------

st.markdown("### Configuración de la Cadena del Enlace")

c1, c2, c3, c4 = st.columns([1,1,1,2])
with c1:
    if st.button("Añadir Fibra", use_container_width=True, type="primary"):
        blk = FiberBlock(type="fiber",
                         par=FiberPar(L=40e3, beta2=-2.1e-26, gamma=1.3e-3, dz=1.0, alpha=4.6e-5)).model_dump()
        ensure_uid(blk); st.session_state.chain.append(blk)
with c2:
    if st.button("Añadir EDFA", use_container_width=True, type="primary"):
        blk = EdfaBlock(type="edfa", par=EdfaPar(G_dB=10.0, nsp=2.5)).model_dump()
        ensure_uid(blk); st.session_state.chain.append(blk)
with c3:
    if st.button("Limpiar Todo", use_container_width=True, type="secondary"):
        st.session_state.chain = []
        st.session_state.edit_idx = None
        st.rerun()
with c4:
    cards_per_row = st.slider("Tarjetas por fila", min_value=1, max_value=3, value=3, help="Ajusta el zoom de la grilla")

columns_grid = st.columns(cards_per_row)
for i, blk in enumerate(st.session_state.chain):
    ensure_uid(blk)
    col = columns_grid[i % cards_per_row]
    with col:
        title = "Fibra" if blk["type"] == "fiber" else "EDFA"
        badge_cls = "fiber" if blk["type"] == "fiber" else "edfa"
        subtitle = summarize_block(blk)

        st.markdown('<div class="mini-card">', unsafe_allow_html=True)
        tcol, bcol = st.columns([0.70, 0.30], gap="small")
        with tcol:
            st.markdown(
                f'<div class="mini-head">'
                f'  <span class="mini-idx">{i+1}</span>'
                f'  <span class="chip {badge_cls}">{title}</span>'
                f'  <span class="mini-sub">{subtitle}</span>'
                f'</div>',
                unsafe_allow_html=True
            )
        with bcol:
            xb1, xb2, xb3, xb4, xb5 = st.columns(5)
            with xb1:
                st.button("▲", key=f"up_{blk['_uid']}", on_click=move_block, args=(i,-1), use_container_width=True)
            with xb2:
                st.button("▼", key=f"down_{blk['_uid']}", on_click=move_block, args=(i,+1), use_container_width=True)
            with xb3:
                st.button("⧉", key=f"dup_{blk['_uid']}", on_click=duplicate_block, args=(i,), use_container_width=True)
            with xb4:
                st.button("✖", key=f"del_{blk['_uid']}", on_click=delete_block, args=(i,), use_container_width=True)
            with xb5:
                def _toggle_edit(k=i):
                    st.session_state.edit_idx = (None if st.session_state.edit_idx == k else k)
                st.button("⚙", key=f"edit_{blk['_uid']}", on_click=_toggle_edit, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

total_m = sum(b["par"]["L"] for b in st.session_state.chain if b["type"] == "fiber")
st.caption(f"Longitud total: {total_m/1e3:.1f} km")

# panel de edición
ei = st.session_state.edit_idx
if ei is not None and 0 <= ei < len(st.session_state.chain):
    blk = st.session_state.chain[ei]
    st.divider()
    st.subheader(f"Editar bloque #{ei+1} - {'Fibra' if blk['type']=='fiber' else 'EDFA'}")

    move_to = st.number_input("Mover a posición", min_value=1, max_value=len(st.session_state.chain), value=ei+1, step=1)
    cpos1, _ = st.columns([0.2, 0.8])
    with cpos1:
        if st.button("Mover", use_container_width=True):
            move_block_to(ei, int(move_to))
            st.session_state.edit_idx = int(move_to) - 1
            st.rerun()

    st.markdown("Parámetros")
    if blk["type"] == "fiber":
        par = blk["par"]
        par["L"]     = st.number_input("L [m]", value=float(par["L"]), min_value=1.0, step=100.0, key=f"L_{blk['_uid']}")
        par["beta2"] = st.number_input("β2 [s^2/m]", value=float(par["beta2"]), step=1e-27, format="%.2e", key=f"b2_{blk['_uid']}")
        par["gamma"] = st.number_input("γ [1/(W·m)]", value=float(par["gamma"]), min_value=1e-6, step=1e-4, format="%.4f", key=f"gm_{blk['_uid']}")
        par["dz"]    = st.number_input("dz [m]", value=float(par["dz"]), min_value=0.1, step=0.1, key=f"dz_{blk['_uid']}")
        par["alpha"] = st.number_input("α [1/m]", value=float(par["alpha"]), min_value=0.0, step=1e-6, format="%.6e", key=f"al_{blk['_uid']}")
    else:
        par = blk["par"]
        par["G_dB"]  = st.number_input("Ganancia [dB]", value=float(par["G_d_B"] if "G_d_B" in par else par["G_dB"]), step=0.1, key=f"G_{blk['_uid']}")
        par["nsp"]   = st.number_input("nsp", value=float(par["nsp"]), min_value=0.5, step=0.1, key=f"nsp_{blk['_uid']}")

    if st.button("Cerrar edición", use_container_width=False):
        st.session_state.edit_idx = None
        st.rerun()

# ------------------------- ejecución -------------------------

st.divider()
st.markdown("### Control de Simulación")

colA, colB = st.columns(2)
with colA: 
    gpu = st.toggle("Usar GPU CuPy", value=True)
with colB: 
    use_dz_override = st.checkbox(
        "Forzar dz global", 
        value=True,  # Activado por defecto
        help="""**Paso SSFM (Split-Step Fourier Method)**: Controla el tamaño del paso espacial usado en la simulación numérica de propagación en fibra óptica.

**¿Qué hace?**
- Sobrescribe el parámetro 'dz' de TODOS los bloques de fibra en la cadena
- Define cada cuántos metros se calculan los efectos lineales (dispersión) y no lineales (SPM, XPM)

**Precisión vs Velocidad:**
- **dz pequeño (1-10 m)**: Alta precisión, captura efectos no lineales sutiles, pero EXTREMADAMENTE lento (horas en enlaces largos)
- **dz medio (10-100 m)**: Balance óptimo para la mayoría de simulaciones
- **dz grande (100-500 m)**: Rápido, adecuado para enlaces cortos o exploración inicial

**Recomendación:** 
- Para enlaces <100 km: 20-50 m
- Para enlaces 100-400 km: 50-100 m  
- Para enlaces >400 km: 100-200 m

**Impacto en tiempo:** Un enlace de 400 km con dz=10 m requiere 40,000 pasos (~30+ min). Con dz=100 m solo 4,000 pasos (~3-5 min)."""
    )
    if use_dz_override:
        dz_override = st.number_input(
            "Paso SSFM [m]", 
            value=50.0,
            min_value=1.0, 
            max_value=2000.0, 
            step=10.0,
            help="Tamaño de paso SSFM. Rápido: 100-500m | Balance: 10-100m | Preciso: 1-10m"
        )
        # Advertencia solo si dz <= 10
        if dz_override <= 10.0:
            st.warning("ADVERTENCIA: dz bajo (≤10 m) puede tardar horas en enlaces largos si no se usa GPU.")
    else:
        dz_override = None

# Paso fijo de captura (no visible para el usuario)
step_const_km = 2.0  # Captura cada 2 km (reducido para optimizar memoria GPU)

col1, col2 = st.columns(2)
with col1: insertion_db = st.number_input("Inserción [dB]", value=1.0, step=0.1)
with col2: splice_db    = st.number_input("Fusión [dB]", value=0.2, step=0.1)

# Plots siempre activados (evita confusión con resultados de ejecuciones previas)
do_const = True
do_eye = True
do_waveform = True

st.number_input("Graficar 2D cada [km]", value=4.0, min_value=2.0, step=2.0,
                key='step_plot2d_km',
                help="Cada cuántos km graficar constelaciones 2D con un mínimo de 2 km. (SI SOLO SE QUIERE TX Y RX USAR UNA DISTANCIA MÁS ALTA QUE EL ENLACE)")

st.markdown("##### Visualización 3D de Constelaciones")

# Calcular valores recomendados según la modulación
M_order = int(M_sel)  # 2, 4, o 16
if M_order == 2:  # BPSK
    recommended_traces = 15
    min_traces = 10
    constellation_regions = 2  # Solo 2 puntos (±1)
elif M_order == 4:  # QPSK
    recommended_traces = 32  # 8 por cuadrante para buena cobertura
    min_traces = 20
    constellation_regions = 4  # 4 cuadrantes
else:  # 16-QAM
    recommended_traces = 64  # 4 por región para 16 puntos
    min_traces = 40
    constellation_regions = 16  # 16 puntos en la constelación

# Mapa de nombres de modulación
mod_map = {2: "BPSK", 4: "QPSK", 16: "16-QAM"}

# Configuración 3D - siempre activado
trace_symbols = True
do_const3d_html = True
step_const3d_km = 0.5  # Siempre usa paso de captura
const3d_html_pts = None  # Se calculará como num_traces

colh4, colh5 = st.columns(2)
with colh4:
    num_traces = st.slider(
        "Número de símbolos a seguir", 
        min_value=min_traces, 
        max_value=200, 
        value=recommended_traces, 
        step=5,
        help=f"Mínimo recomendado para {mod_map[M_order]}: {recommended_traces} símbolos "
             f"(para cubrir las {constellation_regions} regiones de la constelación). "
             f"Menos símbolos pueden dejar regiones sin representar."
    )
    
    # Advertencia si el número es muy bajo
    if num_traces < recommended_traces:
        st.warning(f"ADVERTENCIA: Con {num_traces} símbolos puede que no se cubran todas las regiones "
                  f"de la constelación {mod_map[M_order]} ({constellation_regions} puntos). "
                  f"Recomendado: >={recommended_traces}")

with colh5:
    # Adaptar el texto del agrupamiento según la modulación
    if M_order == 2:  # BPSK
        group_label = "Agrupar por fase (±1)"
        group_help = "Agrupa símbolos por su fase (positivo/negativo) con colores diferentes."
    elif M_order == 4:  # QPSK
        group_label = "Agrupar por cuadrante QPSK"
        group_help = "Agrupa símbolos por cuadrante (4 regiones) con colores diferentes para facilitar interpretación."
    else:  # 16-QAM
        group_label = "Agrupar por región 16-QAM"
        group_help = "Agrupa símbolos por su región en la constelación 16-QAM (16 puntos agrupados por cercanía)."
    
    group_by_quadrant = st.checkbox(group_label, value=True, help=group_help)

# Carpetas fijas para plots y logs (simplificación de interfaz)
plots_dir = "plots"
outdir = "logs"

# ========== VALIDACIÓN OPCIONAL DE PARÁMETROS (COLAPSABLE) ==========
with st.expander("Validar Configuración (opcional)", expanded=False):
    st.caption("Validación básica de parámetros físicos antes de ejecutar. "
              "Son sugerencias, no restricciones. El simulador es un playground experimental.")
    
    # Extraer parámetros globales
    gp = st.session_state["global"]
    Rb = float(gp.get("bit_rate", 1e9))
    M_order = int(gp.get("M", 4))
    sps = int(gp.get("sps", 8))
    Fs = float(gp.get("Fs", Rb * sps / np.log2(M_order)))
    Ptx_W = float(gp.get("Ptx", 1.0))  # Potencia en Watts
    Ptx_mW = Ptx_W * 1000  # Convertir a mW
    Ptx_dBm = 10 * np.log10(Ptx_mW)

    Rs = Rb / np.log2(M_order)  # Symbol rate

    # Calcular longitud total y DCF ratio
    total_fiber_length_km = 0
    total_smf_length_km = 0
    total_dcf_length_km = 0
    total_disp_smf = 0
    total_disp_dcf = 0

    for blk in st.session_state.chain:
        if blk["type"] == "fiber":
            L_km = blk["par"]["L"] / 1000.0
            total_fiber_length_km += L_km
            
            # Detectar tipo de fibra por dispersión
            D = blk["par"].get("D", 0)
            if D > 0:  # SMF
                total_smf_length_km += L_km
                total_disp_smf += D * L_km
            elif D < 0:  # DCF
                total_dcf_length_km += L_km
                total_disp_dcf += abs(D * L_km)

    # Calcular DCF compensation ratio
    if total_disp_smf > 0 and total_disp_dcf > 0:
        dcf_ratio = total_disp_dcf / total_disp_smf
    else:
        dcf_ratio = 0

    # Extraer gamma y nsp promedio
    gamma_avg = 0
    nsp_avg = 0
    num_fibers = 0
    num_edfas = 0

    for blk in st.session_state.chain:
        if blk["type"] == "fiber":
            gamma_avg += abs(blk["par"].get("gamma", 0))
            num_fibers += 1
        elif blk["type"] == "edfa":
            nsp_avg += blk["par"].get("nsp", 1.0)
            num_edfas += 1

    if num_fibers > 0:
        gamma_avg /= num_fibers
    if num_edfas > 0:
        nsp_avg /= num_edfas

    # Validaciones
    validation_warnings = []
    validation_errors = []
    validation_ok = []

    # 1. Validación de Ptx
    if Ptx_dBm < -10:
        validation_errors.append(f"Potencia de transmisión muy baja: {Ptx_dBm:.1f} dBm ({Ptx_mW:.3f} mW). "
                                f"El enlace estará dominado por ruido térmico/ASE. Mínimo recomendado: 0 dBm (1 mW).")
    elif Ptx_dBm < 0:
        validation_warnings.append(f"Potencia de transmisión baja: {Ptx_dBm:.1f} dBm ({Ptx_mW:.3f} mW). "
                                  f"Puede tener alto BER. Recomendado: 0-10 dBm.")
    else:
        validation_ok.append(f"Potencia de transmisión: {Ptx_dBm:.1f} dBm ({Ptx_mW:.1f} mW)")

    # 2. Validación de Fs
    # Nota: Este simulador usa Fs = Rb × sps (oversampling intencional)
    # Esto proporciona mayor resolución en DSP que el mínimo teórico Fs = Rs × sps
    Fs_min_theoretical = Rs * sps  # Mínimo teórico (Nyquist en símbolos)
    
    if Fs < Fs_min_theoretical * 0.95:  # Fs muy bajo: riesgo de aliasing
        validation_errors.append(f"Frecuencia de muestreo muy baja: Fs={Fs/1e9:.3f} GS/s < mínimo={Fs_min_theoretical/1e9:.3f} GS/s. "
                                f"RIESGO DE ALIASING.")
    else:
        oversampling_factor = Fs / Fs_min_theoretical
        validation_ok.append(f"Frecuencia de muestreo: {Fs/1e9:.3f} GS/s ({oversampling_factor:.1f}× oversampling intencional)")

    # 3. Validación de DCF compensation
    if total_dcf_length_km > 0:
        if 0.95 <= dcf_ratio <= 1.05:
            validation_ok.append(f"Compensación DCF perfecta: ratio={dcf_ratio:.3f} (dispersión residual ~0)")
        elif 0.80 <= dcf_ratio <= 1.20:
            validation_warnings.append(f"Compensación DCF parcial: ratio={dcf_ratio:.3f}. "
                                      f"Habrá dispersión residual. Óptimo: 0.95-1.05")
        else:
            validation_errors.append(f"Compensación DCF muy desbalanceada: ratio={dcf_ratio:.3f}. "
                                    f"SMF: {total_disp_smf:.1f} ps/nm, DCF: {total_disp_dcf:.1f} ps/nm. "
                                    f"Ratio óptimo: ~1.0 (típicamente 5:1 en longitud SMF:DCF)")

    # 4. Validación de efectos no lineales
    if gamma_avg > 1e-5:  # Si hay gamma significativo (>0.01 /W/km)
        # Advertir si Ptx >= 5 dBm (3.16 mW) con enlaces largos
        if Ptx_dBm >= 10:  # Potencia muy alta
            validation_warnings.append(f"Potencia de lanzamiento alta: Ptx={Ptx_dBm:.1f} dBm con gamma={gamma_avg:.3e} /W/m. "
                                      f"Efectos no lineales (SPM/XPM/FWM) serán significativos y degradarán la señal. "
                                      f"Considere reducir Ptx a 0-5 dBm para enlaces largos.")
        elif Ptx_dBm >= 5:  # Potencia moderada-alta
            validation_warnings.append(f"Potencia de lanzamiento moderada-alta: Ptx={Ptx_dBm:.1f} dBm con gamma={gamma_avg:.3e} /W/m. "
                                      f"Pueden aparecer efectos no lineales (SPM/XPM). "
                                      f"Monitoree la calidad de la constelación.")
    else:
        # Si gamma ≈ 0, la potencia alta no es problema (fibra ideal)
        if Ptx_dBm >= 15:
            validation_warnings.append(f"Potencia muy alta: Ptx={Ptx_dBm:.1f} dBm. "
                                      f"Aunque gamma≈0, verifique que los amplificadores no saturen.")

    # 5. Validación de ASE noise
    if nsp_avg > 2.5:
        validation_warnings.append(f"Factor de ruido alto: nsp={nsp_avg:.2f}. "
                                  f"EDFAs con ruido elevado, BER estará limitado por ASE. Típico: 1.5-2.0")
    else:
        validation_ok.append(f"Factor de ruido EDFA: nsp={nsp_avg:.2f} (aceptable)")

    # 6. Validación de longitud del enlace
    if total_fiber_length_km > 1000:
        validation_warnings.append(f"Enlace muy largo: {total_fiber_length_km:.0f} km. "
                                  f"La simulación puede tardar horas y usar mucha memoria GPU.")
    elif total_fiber_length_km > 500:
        validation_warnings.append(f"Enlace largo: {total_fiber_length_km:.0f} km. "
                                  f"Tiempo de simulación elevado.")
    else:
        validation_ok.append(f"Longitud de enlace: {total_fiber_length_km:.1f} km")

    # 7. Validación de dz vs longitud (si está forzado)
    if use_dz_override and dz_override is not None:
        num_steps = (total_fiber_length_km * 1000) / dz_override
        if num_steps > 100000:
            validation_errors.append(f"Paso SSFM muy pequeño: dz={dz_override:.1f} m requiere {num_steps:.0f} pasos "
                                    f"para {total_fiber_length_km:.0f} km. Esto puede tardar HORAS. "
                                    f"Recomendado: dz >= {total_fiber_length_km * 1000 / 50000:.0f} m (50k pasos)")
        elif num_steps > 50000:
            validation_warnings.append(f"Paso SSFM pequeño: dz={dz_override:.1f} m requiere {num_steps:.0f} pasos. "
                                      f"Simulación lenta (>30 min posible).")
        else:
            validation_ok.append(f"Paso SSFM: dz={dz_override:.1f} m ({num_steps:.0f} pasos, tiempo razonable)")

    # Mostrar resultados de validación
    if validation_errors:
        st.error("Sugerencias críticas:")
        for err in validation_errors:
            st.error(f"• {err}")

    if validation_warnings:
        st.warning("Advertencias:")
        for warn in validation_warnings:
            st.warning(f"• {warn}")

    if validation_ok:
        st.success("Parámetros dentro de rangos típicos:")
        for ok in validation_ok:
            st.success(f"• {ok}")
    
    if not validation_errors and not validation_warnings and not validation_ok:
        st.info("Ejecuta la validación expandiendo esta sección.")

st.markdown("---")

if st.button("Ejecutar simulación", type="primary"):
    try:
        # Agregar session_id al config global antes de validar
        global_with_session = dict(st.session_state["global"])
        global_with_session["session_id"] = st.session_state.session_id
        global_with_session["plots_dir"] = plots_dir
        
        cfg_dict = {
            "global": global_with_session,
            "pulse":  st.session_state["pulse"],
            "chain":  st.session_state.chain,
        }
        cfg_model = SimConfig.model_validate(cfg_dict)
        cfg_norm  = cfg_model.model_dump(by_alias=True)
    except Exception as e:
        st.error(f"Config inválida: {e}")
    else:
        tmp = pathlib.Path("tmp_gui_config.json")
        tmp.write_text(json.dumps(cfg_norm, indent=2), encoding="utf-8")

        try:
            backend_info = _execute(
                config=str(tmp),
                outdir=outdir,
                gpu=bool(gpu),
                dz=float(dz_override) if dz_override is not None else None,
                insertion_db=float(insertion_db),
                use_insertion_loss=True,
                splice_db=float(splice_db),
                use_splice_loss=True,
                do_const=bool(do_const),
                step_const_km=float(step_const_km),
                step_plot2d_km=float(st.session_state.get('step_plot2d_km', 5.0)),
                do_eye=bool(do_eye),
                plots_dir=plots_dir,
                do_const3d=False, const3d_every=1, const3d_pts=1000,
                do_const3d_html=bool(do_const3d_html),
                step_const3d_km=float(step_const3d_km),
                const3d_html_pts=int(num_traces if trace_symbols else const3d_html_pts),
                trace_symbols=bool(trace_symbols),
                num_traces=int(num_traces if trace_symbols else 50),
                group_by_quadrant=bool(group_by_quadrant),
                do_waveform=bool(do_waveform),
            )
            st.session_state["last_backend"] = backend_info
            st.success("Simulación terminada")
        except Exception as e:
            st.error("Fallo durante la simulación")
            st.exception(e)

# ------------------------- resumen y resultados -------------------------

def _read_last_log(outdir: str) -> Dict[str, Any] | None:
    p = pathlib.Path(outdir)
    if not p.exists(): return None
    cands = sorted(p.glob("simlog_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    if not cands: return None
    try:
        return json.loads(cands[0].read_text(encoding="utf-8"))
    except Exception:
        return None

_h = 6.62607015e-34
_c = 299792458.0

def _W_to_dBm(p_W: float) -> float:
    return 10.0 * math.log10(max(p_W, 1e-30) / 1e-3)

def build_profile_from_state(Bo_Hz: float = 12.5e9) -> List[Dict[str, Any]]:
    g = st.session_state["global"]; chain = st.session_state.chain
    lambda_nm = float(g.get("lambda_nm", 1550.0))
    nu = _c / (lambda_nm * 1e-9)
    z_m = 0.0
    P_sig_W = float(g["Ptx"])
    P_ase_W = 0.0
    prof: List[Dict[str, Any]] = []
    for i, blk in enumerate(chain):
        t = blk["type"]; par = blk["par"]
        if t == "fiber":
            L = float(par["L"]); alpha = float(par["alpha"])
            P_sig_W *= math.exp(-alpha * L); z_m += L
            prof.append({"i": i, "kind": "fiber", "z_km": z_m/1e3, "P_dBm": _W_to_dBm(P_sig_W),
                         "OSNR_dB": None if P_ase_W <= 0 else 10*math.log10(P_sig_W/max(P_ase_W,1e-30))})
        elif t == "edfa":
            G_dB = float(par.get("G_dB", 0.0)); G = 10.0**(G_dB/10.0); nsp = float(par.get("nsp", 2.5))
            P_sig_W *= G
            P_ase_W += 2.0 * nsp * _h * nu * (G - 1.0) * Bo_Hz
            prof.append({"i": i, "kind": "edfa", "z_km": z_m/1e3, "P_dBm": _W_to_dBm(P_sig_W),
                         "OSNR_dB": 10*math.log10(P_sig_W/max(P_ase_W,1e-30)), "G_dB": G_dB, "nsp": nsp})
        else:
            prof.append({"i": i, "kind": t, "z_km": z_m/1e3, "P_dBm": _W_to_dBm(P_sig_W),
                         "OSNR_dB": None if P_ase_W <= 0 else 10*math.log10(P_sig_W/max(P_ase_W,1e-30))})
    
    # BER aproximada por punto (BPSK) - NOTA: Solo considera ruido ASE
    # Esta es una estimación teórica que NO incluye efectos no lineales,
    # dispersión, timing jitter, etc. Para BER real, usar SNR medido post-DSP
    from math import erfc, sqrt
    Rb = float(g["Rb"])
    for pt in prof:
        osnr = pt.get("OSNR_dB", None)
        if osnr is None: 
            pt["BER_theoretical"] = None
            continue
        
        # Conversión OSNR → SNR teórico (puede sobreestimar performance)
        OSNR_lin = 10.0**(osnr/10.0)
        SNR_lin = OSNR_lin * (Bo_Hz / max(Rb, 1.0))
        pt["BER_theoretical"] = 0.5 * erfc(sqrt(max(SNR_lin,1e-12))/sqrt(2.0))
        
        # Mantener compatibilidad
        pt["BER"] = pt["BER_theoretical"]
    return prof

def _profile_to_csv_bytes(profile: List[Dict[str, Any]]) -> bytes:
    import csv
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["idx","tipo","z_km","P_dBm","OSNR_dB","BER","G_dB","nsp"])
    for p in profile:
        w.writerow([p.get("i"), p.get("kind"), p.get("z_km"), p.get("P_dBm"),
                    p.get("OSNR_dB"), p.get("BER"), p.get("G_dB"), p.get("nsp")])
    return buf.getvalue().encode("utf-8")

plots_p = pathlib.Path(plots_dir)
session_id = st.session_state.session_id
png1 = plots_p / f"constelaciones_{session_id}.png"
png2 = plots_p / f"potencia_{session_id}.png"
eye  = plots_p / f"eye_{session_id}.png"

st.divider()
st.markdown("### Resultados de la Simulación")

# ---------- Resumen superior (chips + métricas) ----------
log = _read_last_log(outdir)
res = (log or {}).get("result", {}) if log else {}

# Meta chips (texto más chico, con wrap)
chips = []
bk = st.session_state.get("last_backend") or res.get("backend")
if bk: chips.append(f"<span class='meta-chip'>Backend: {bk}</span>")

# Obtener métricas unificadas (nueva estructura)
ber = res.get("BER", None)  # BER unificado
snr_sym = res.get("SNR_sym_dB", None)  # SNR medido post-DSP
osnr = res.get("OSNR_final_dB", None)  # OSNR óptico

# Mostrar métricas relevantes
if ber is not None:
    chips.append(f"<span class='meta-chip'>BER: {float(ber)*100:.4f}%</span>")

if snr_sym is not None:
    chips.append(f"<span class='meta-chip'>SNR medido (post-DSP): {snr_sym:.2f} dB</span>")

if chips:
    st.markdown("<div class='meta-row'>" + "".join(chips) + "</div>", unsafe_allow_html=True)

with st.expander("Resumen de la última ejecución", expanded=True):
    if log:
        cols = st.columns(7)  # 7 columnas: Backend, Tiempo, OSNR, SNR pre-DSP, SNR post-DSP, Pout, BER

        # Backend abreviado para que no se corte
        bk_full = res.get("backend", "")
        if "CuPy" in bk_full:
            bk_short = "GPU CuPy"
        elif "NumPy" in bk_full:
            bk_short = "CPU NumPy"
        else:
            bk_short = bk_full[:16] + ("…" if len(bk_full) > 16 else "")

        cols[0].metric(
            "Backend", 
            bk_short,
            help="Motor de cálculo utilizado: GPU CuPy (aceleración por GPU, hasta 100× más rápido) o CPU NumPy (solo procesador, más lento pero sin requerir CUDA)."
        )
        cols[1].metric(
            "Tiempo [s]", 
            f"{res.get('elapsed_s', 0):.3f}",
            help="Tiempo total de ejecución de la simulación en segundos. Incluye propagación SSFM, procesamiento DSP y generación de gráficos."
        )

        # OSNR (solo ruido óptico ASE)
        osnr = res.get("OSNR_final_dB", None)
        
        if osnr is not None:
            cols[2].metric(
                "OSNR (ASE) [dB]", 
                f"{osnr:.2f}",
                help="Optical Signal-to-Noise Ratio al final del enlace. Mide solo el ruido ASE de los EDFAs (ruido óptico). Valores típicos: >20 dB excelente, 15-20 dB bueno, <15 dB degradado. No incluye ruido eléctrico (AWGN)."
            )
        else:
            cols[2].metric(
                "OSNR", 
                "n/a",
                help="Optical Signal-to-Noise Ratio al final del enlace. No disponible para esta simulación."
            )

        # SNR pre-DSP (medido en waveform antes del matched filter)
        snr_pre = res.get("SNR_pre_dsp_dB", None)
        if snr_pre is not None:
            cols[3].metric(
                "SNR pre-DSP [dB]",
                f"{snr_pre:.2f}",
                help="SNR medido directamente en la waveform recibida ANTES del matched filter. Refleja la calidad de la señal tal como llega al receptor, incluyendo todos los ruidos (ASE, AWGN TX/RX, no linealidades, dispersión). Este es el SNR 'crudo' sin procesamiento DSP."
            )
        else:
            cols[3].metric(
                "SNR pre-DSP",
                "n/a",
                help="SNR pre-DSP no disponible."
            )

        # SNR medido post-DSP
        snr_sym = res.get("SNR_sym_dB", None)
        if snr_sym is not None:
            cols[4].metric(
                "SNR post-DSP [dB]",
                f"{snr_sym:.2f}",
                help="SNR medido después del procesamiento DSP completo (sincronización, CPR, matched filter). Incluye todos los ruidos: ASE, AWGN TX/RX, efectos no lineales, dispersión residual. Este es el SNR real que 've' el demodulador en los símbolos recuperados. Valores típicos: >15 dB excelente, 10-15 dB bueno, 5-10 dB degradado, <5 dB crítico."
            )
        else:
            cols[4].metric(
                "SNR post-DSP",
                "n/a",
                help="SNR medido post-DSP no disponible."
            )

        pout = res.get("Pout_dBm", None)
        cols[5].metric(
            "Pout [dBm]", 
            f"{pout:.2f}" if pout is not None else "n/a",
            help="Potencia óptica de salida al final del enlace en dBm. Indica cuánta potencia llega al receptor después de todas las pérdidas y ganancias. Valores típicos: -10 a -25 dBm. Si es muy baja (<-30 dBm), el receptor tendrá problemas de sensibilidad."
        )
        
        # BER unificado (para todas las modulaciones)
        ber = res.get("BER", None)
        cols[6].metric(
            "BER", 
            f"{ber:.3e}" if ber is not None else "n/a",
            help="Bit Error Rate medido con el método mejorado de demodulación (sincronización temporal, CPR, normalización de ganancia). Fracción de bits decodificados incorrectamente. Valores típicos: <1e-12 excelente, 1e-9 a 1e-12 muy bueno, 1e-6 a 1e-9 bueno (con FEC), 1e-3 a 1e-6 límite FEC, >1e-3 inutilizable. Válido para BPSK, QPSK y 16-QAM."
        )
    else:
        st.info("Aún no hay logs en la carpeta seleccionada.")

# 3D interactivo (HTML) y PNGs existentes
cands = sorted(plots_p.glob("constelaciones_3d*.html"), key=lambda p: p.stat().st_mtime, reverse=True)
if cands:
    st.markdown("### Evolución 3D de constelaciones a lo largo del enlace")
    html = cands[0].read_text(encoding="utf-8")
    components.html(html, height=800, scrolling=False)

# Sección de gráficos principales - organizados horizontalmente con mismo tamaño
st.divider()
st.markdown("### Gráficos de Resultados")

col1, col2, col3 = st.columns(3)
with col1:
    if png1.exists(): 
        st.image(str(png1), caption="Constelaciones (grid)", use_container_width=True)
with col2:
    if png2.exists(): 
        st.image(str(png2), caption="Evolución de potencia", use_container_width=True)
with col3:
    if eye.exists():  
        st.image(str(eye), caption="Eye Diagram", use_container_width=True)

# Waveforms TX/RX si existen
waveform_plot = plots_p / f"waveform_comparison_{session_id}.png"
waveform_h5 = plots_p / f"waveforms_{session_id}.h5"
if waveform_plot.exists():
    st.divider()
    st.markdown("### Waveforms TX vs RX")
    st.image(str(waveform_plot), caption="Comparación de señales: Transmitida (TX) vs Recibida (RX)", use_container_width=True)
    
    # Botones de descarga por separado: TX, RX, RX post-MF
    if waveform_h5.exists():
        try:
            import h5py
            import io
            import datetime
            
            # Generar timestamp y metadata para nombres descriptivos únicos
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            mod_name = res.get("global", {}).get("mod", "BPSK")
            Rb_Gbps = res.get("global", {}).get("Rb", 32e9) / 1e9
            Ptx_W = res.get("global", {}).get("Ptx", 0.001)
            Ptx_dBm = 10 * np.log10(Ptx_W * 1000) if Ptx_W > 0 else -30
            L_total_km = res.get("Lcum_m", 0.0) / 1000.0
            
            with h5py.File(waveform_h5, 'r') as f:
                tx_real = f['tx/real'][:]
                tx_imag = f['tx/imag'][:]
                rx_real = f['rx/real'][:]
                rx_imag = f['rx/imag'][:]
                
                # Metadata común
                Fs = f.attrs.get('Fs', 'N/A')
                sps = f.attrs.get('sps', 'N/A')
                mod = f.attrs.get('mod', 'N/A')
                Ptx = f.attrs.get('Ptx', 'N/A')
                
                # Función helper para generar CSV
                def make_csv(data_real, data_imag, title):
                    buffer = io.StringIO()
                    buffer.write(f"# {title} - FiberSim\n")
                    buffer.write(f"# Timestamp: {timestamp}\n")
                    buffer.write(f"# Fs = {Fs} Hz\n")
                    buffer.write(f"# sps = {sps}\n")
                    buffer.write(f"# Modulacion = {mod}\n")
                    if Ptx != 'N/A':
                        buffer.write(f"# Ptx = {Ptx} W\n")
                    buffer.write("# Columnas: sample_idx, real, imag, magnitude\n")
                    buffer.write("#\n")
                    
                    for i in range(len(data_real)):
                        r = data_real[i]
                        im = data_imag[i]
                        mag = (r**2 + im**2)**0.5
                        buffer.write(f"{i},{r:.6e},{im:.6e},{mag:.6e}\n")
                    
                    return buffer.getvalue()
                
                # CSV para TX
                csv_tx = make_csv(tx_real, tx_imag, "Waveform TX (Transmitido)")
                
                # CSV para RX
                csv_rx = make_csv(rx_real, rx_imag, "Waveform RX (Recibido pre-DSP)")
                
                # CSV para RX post-MF (si existe)
                csv_rx_post = None
                if 'rx_post_mf/real' in f and 'rx_post_mf/imag' in f:
                    rx_post_real = f['rx_post_mf/real'][:]
                    rx_post_imag = f['rx_post_mf/imag'][:]
                    csv_rx_post = make_csv(rx_post_real, rx_post_imag, "Waveform RX (Post Matched Filter)")
            
            # Botones de descarga en 3 columnas
            col_tx, col_rx, col_rx_post = st.columns(3)
            
            with col_tx:
                st.download_button(
                    label="Descargar TX",
                    data=csv_tx,
                    file_name=f"TX_{mod_name}_{Ptx_dBm:.1f}dBm_{L_total_km:.0f}km_{timestamp}.csv",
                    mime="text/csv",
                    help="Señal transmitida (después de pulse shaping)",
                    use_container_width=True
                )
            
            with col_rx:
                st.download_button(
                    label="Descargar RX (pre-DSP)",
                    data=csv_rx,
                    file_name=f"RX_preDSP_{mod_name}_{Ptx_dBm:.1f}dBm_{L_total_km:.0f}km_{timestamp}.csv",
                    mime="text/csv",
                    help="Señal recibida después de propagación óptica (antes de matched filter)",
                    use_container_width=True
                )
            
            with col_rx_post:
                if csv_rx_post:
                    st.download_button(
                        label="Descargar RX (post-MF)",
                        data=csv_rx_post,
                        file_name=f"RX_postMF_{mod_name}_{Ptx_dBm:.1f}dBm_{L_total_km:.0f}km_{timestamp}.csv",
                        mime="text/csv",
                        help="Señal recibida después de matched filter RRC",
                        use_container_width=True
                    )
                else:
                    st.caption("Post-MF no disponible")
                    
        except Exception as e:
            st.warning(f"No se pudieron generar archivos de descarga: {e}")

# Perfiles z: preferir medidos del último log; si no, estimado analítico
st.divider()
is_measured = False
profile_res = res.get("profile", None)
profile: List[Dict[str, Any]]
if isinstance(profile_res, list) and profile_res:
    profile = profile_res
    is_measured = True
    st.markdown("### Análisis del Perfil del Enlace")
else:
    st.markdown("### Análisis del Perfil del Enlace (Estimado)")

# Controles: mostrar OSNR/BER solo si hay datos disponibles en el perfil actual
has_osnr = any(p.get("OSNR_dB") is not None for p in (profile_res or [])) if is_measured else True
has_ber  = any(p.get("BER") is not None for p in (profile_res or [])) if is_measured else True

if has_osnr and has_ber:
    pc1, pc2, pc3 = st.columns(3)
    with pc1: show_P = st.checkbox("Potencia [dBm]", value=True)
    with pc2: show_O = st.checkbox("OSNR [dB]", value=True)
    with pc3: show_B = st.checkbox("BER", value=False)
elif has_osnr and not has_ber:
    pc1, pc2 = st.columns(2)
    with pc1: show_P = st.checkbox("Potencia [dBm]", value=True)
    with pc2: show_O = st.checkbox("OSNR [dB]", value=True)
    show_B = False
elif (not has_osnr) and has_ber:
    pc1, pc2 = st.columns(2)
    with pc1: show_P = st.checkbox("Potencia [dBm]", value=True)
    with pc2: show_B = st.checkbox("BER", value=False)
    show_O = False
else:
    # Solo potencia
    show_O = False; show_B = False
    show_P = st.checkbox("Potencia [dBm]", value=True)

# Bo solo aplica al perfil estimado
if not is_measured:
    Bo_GHz = st.number_input("Bo [GHz] para OSNR", value=12.5, min_value=0.1, max_value=100.0, step=0.1)
    profile = build_profile_from_state(Bo_Hz=float(Bo_GHz)*1e9)
    st.info("Mostrando perfil estimado localmente en GUI (no hay perfil medido en el último log).")

try:
    # Identificar posiciones de EDFAs (saltos de potencia > 5 dB)
    edfa_positions = []
    for i in range(1, len(profile)):
        p_before = profile[i-1].get("P_dBm")
        p_after = profile[i].get("P_dBm")
        if p_before is not None and p_after is not None:
            delta_p = p_after - p_before
            if delta_p > 5.0:  # Ganancia significativa = EDFA
                z_edfa = profile[i]["z_km"]
                edfa_positions.append((z_edfa, delta_p))
    
    traces = []
    z = [p["z_km"] for p in profile]
    
    if show_P:
        traces.append(go.Scatter(
            x=z, y=[p.get("P_dBm") for p in profile], 
            mode="lines+markers", 
            name="Potencia [dBm]",
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=4)
        ))
    if show_O:
        osnr_values = [p.get("OSNR_dB") for p in profile]
        traces.append(go.Scatter(
            x=z, y=osnr_values, 
            mode="lines+markers", 
            name="OSNR [dB]",
            line=dict(color='#ff7f0e', width=2),
            marker=dict(size=4)
        ))
    
    layout = go.Layout(
        xaxis=dict(title="Distancia [km]", gridcolor='#e0e0e0'),
        yaxis=dict(title="dB / dBm", gridcolor='#e0e0e0'),
        yaxis2=dict(title="BER", overlaying="y", side="right", type="log") if show_B else None,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        title="Evolución de Potencia y OSNR a lo largo del enlace",
        hovermode='x unified',
    )
    fig = go.Figure(data=traces, layout=layout)
    
    if show_B:
        fig.add_trace(go.Scatter(
            x=z, y=[p.get("BER") for p in profile], 
            mode="lines+markers", 
            name="BER", 
            yaxis="y2",
            line=dict(color='#d62728', width=2),
            marker=dict(size=4)
        ))
    
    # Añadir marcadores de EDFAs
    for z_edfa, gain_db in edfa_positions:
        fig.add_vline(
            x=z_edfa, 
            line_dash="dash", 
            line_color="green", 
            opacity=0.5,
            annotation_text=f"EDFA (+{gain_db:.1f}dB)",
            annotation_position="top"
        )
    
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
    
    # Información adicional sobre el perfil
    if edfa_positions:
        st.caption(f"{len(edfa_positions)} amplificadores ópticos detectados en el enlace (líneas verdes punteadas)")
    
except Exception as e:
    st.error(f"Error generando gráficos de perfil: {e}")
    st.line_chart({"z_km": [p["z_km"] for p in profile],
                   "P_dBm": [p.get("P_dBm") for p in profile]})

col_dl1, col_dl2 = st.columns(2)
with col_dl1:
    st.download_button("Descargar perfil CSV",
                       data=_profile_to_csv_bytes(profile),
                       file_name="perfil.csv", mime="text/csv")
with col_dl2:
    st.download_button("Descargar perfil JSON",
                       data=json.dumps(profile, indent=2).encode("utf-8"),
                       file_name="perfil.json", mime="application/json")

