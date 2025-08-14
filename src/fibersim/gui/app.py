# src/fibersim/gui/app.py
from __future__ import annotations
import json, pathlib, uuid, math, io
from typing import Any, Dict, List
import streamlit as st
import streamlit.components.v1 as components

# Modelos y ejecución
from fibersim.schema import SimConfig, FiberBlock, EdfaBlock, FiberPar, EdfaPar
from fibersim.main import _execute

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

# ------------------------- presets -------------------------

PRESETS: Dict[str, Dict[str, Any]] = {
    "Demo 400 km DCF": {
        "global": { "project": "Enlace 400 km con DCF", "Rb": 10e9, "M": 2, "sps": 8, "Fs": 80e9, "Nsym": 32768, "Ptx": 3.162e-5, "lambda_nm": 1550 },
        "pulse":  { "type": "RRC", "roll": 0.2, "span": 8 },
        "chain": [
            { "type": "fiber", "par": { "L": 66000.0, "beta2": -2.127e-26, "gamma": 0.0013, "dz": 1000.0, "alpha": 4.605e-5 } },
            { "type": "fiber", "par": { "L": 14000.0, "beta2":  1.001e-25, "gamma": 0.0020, "dz": 1000.0, "alpha": 1.151e-4 } },
            { "type": "edfa",  "par": { "G_dB": 21.2, "nsp": 1.6 } },

            { "type": "fiber", "par": { "L": 66000.0, "beta2": -2.127e-26, "gamma": 0.0013, "dz": 1000.0, "alpha": 4.605e-5 } },
            { "type": "fiber", "par": { "L": 14000.0, "beta2":  1.001e-25, "gamma": 0.0020, "dz": 1000.0, "alpha": 1.151e-4 } },
            { "type": "edfa",  "par": { "G_dB": 21.2, "nsp": 1.6 } },

            { "type": "fiber", "par": { "L": 66000.0, "beta2": -2.127e-26, "gamma": 0.0013, "dz": 1000.0, "alpha": 4.605e-5 } },
            { "type": "fiber", "par": { "L": 14000.0, "beta2":  1.001e-25, "gamma": 0.0020, "dz": 1000.0, "alpha": 1.151e-4 } },
            { "type": "edfa",  "par": { "G_dB": 21.2, "nsp": 1.6 } },

            { "type": "fiber", "par": { "L": 66000.0, "beta2": -2.127e-26, "gamma": 0.0013, "dz": 1000.0, "alpha": 4.605e-5 } },
            { "type": "fiber", "par": { "L": 14000.0, "beta2":  1.001e-25, "gamma": 0.0020, "dz": 1000.0, "alpha": 1.151e-4 } },
            { "type": "edfa",  "par": { "G_dB": 21.2, "nsp": 1.6 } },

            { "type": "fiber", "par": { "L": 66000.0, "beta2": -2.127e-26, "gamma": 0.0013, "dz": 1000.0, "alpha": 4.605e-5 } },
            { "type": "fiber", "par": { "L": 14000.0, "beta2":  1.001e-25, "gamma": 0.0020, "dz": 1000.0, "alpha": 1.151e-4 } },
            { "type": "edfa",  "par": { "G_dB": 21.2, "nsp": 1.6 } }
        ]
    },
    "Demo 80 km SMF": {
        "global": { "project": "Enlace 80 km SMF", "Rb": 10e9, "M": 2, "sps": 8, "Fs": 80e9, "Nsym": 16384, "Ptx": 1e-3, "lambda_nm": 1550 },
        "pulse":  { "type": "RRC", "roll": 0.25, "span": 6 },
        "chain":  [
            { "type": "fiber", "par": { "L": 80000.0, "beta2": -2.127e-26, "gamma": 0.0013, "dz": 1000.0, "alpha": 4.605e-5 } }
        ]
    },
    "Demo 3 spans SMF+EDFA": {
        "global": { "project": "3 spans SMF+EDFA", "Rb": 10e9, "M": 2, "sps": 8, "Fs": 80e9, "Nsym": 16384, "Ptx": 1e-3, "lambda_nm": 1550 },
        "pulse":  { "type": "RRC", "roll": 0.25, "span": 6 },
        "chain": [
            { "type": "fiber", "par": { "L": 50000.0, "beta2": -2.127e-26, "gamma": 0.0013, "dz": 1000.0, "alpha": 4.605e-5 } },
            { "type": "edfa",  "par": { "G_dB": 10.0, "nsp": 1.6 } },
            { "type": "fiber", "par": { "L": 50000.0, "beta2": -2.127e-26, "gamma": 0.0013, "dz": 1000.0, "alpha": 4.605e-5 } },
            { "type": "edfa",  "par": { "G_dB": 10.0, "nsp": 1.6 } },
            { "type": "fiber", "par": { "L": 50000.0, "beta2": -2.127e-26, "gamma": 0.0013, "dz": 1000.0, "alpha": 4.605e-5 } }
        ]
    }
}

# ------------------------- página y estado -------------------------

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
</style>
""", unsafe_allow_html=True)

if "chain" not in st.session_state:
    st.session_state.chain: List[dict] = []
if "global" not in st.session_state:
    st.session_state["global"] = dict(Rb=32e9, M=2, sps=32, Fs=1.024e12, Nsym=4096, Ptx=1e-2, lambda_nm=1550)
if "pulse" not in st.session_state:
    st.session_state["pulse"] = dict(type="RRC", roll=0.1, span=10)
if "edit_idx" not in st.session_state:
    st.session_state.edit_idx: int | None = None

# ------------------------- carga, guardado y presets -------------------------

st.sidebar.header("⚙️ Configuración")
upl = st.sidebar.file_uploader("Cargar JSON", type=["json"])
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

def export_json() -> str:
    data = {"global": st.session_state["global"], "pulse": st.session_state["pulse"], "chain": st.session_state.chain}
    return json.dumps(data, indent=2)

st.sidebar.download_button("💾 Descargar JSON", data=export_json(), file_name="config.json", mime="application/json")

st.sidebar.markdown("---")
st.sidebar.subheader("📁 Ejemplos")
preset_sel = st.sidebar.selectbox("Elegir ejemplo", options=list(PRESETS.keys()), index=0)
if st.sidebar.button("Cargar ejemplo", use_container_width=True):
    try:
        raw = PRESETS[preset_sel]
        cfg = SimConfig.model_validate(raw)
        st.session_state["global"] = cfg.global_.model_dump()
        st.session_state["pulse"]  = cfg.pulse.model_dump()
        st.session_state.chain     = [b.model_dump() for b in cfg.chain]
        for b in st.session_state.chain: ensure_uid(b)
        st.session_state.edit_idx = None
        st.sidebar.success("Ejemplo cargado")
    except Exception as e:
        st.sidebar.error(f"No se pudo cargar: {e}")

# ------------------------- parámetros globales y pulso -------------------------

st.title("FiberSim - Construcción de enlace")

gcol, pcol = st.columns(2)
with gcol:
    st.subheader("Parámetros Globales")
    g = st.session_state["global"]
    g["Rb"]   = st.number_input("Rb [baud]", value=float(g["Rb"]), min_value=1e6, step=1e6, format="%.0f")
    g["M"]    = st.number_input("Orden M", value=int(g["M"]), min_value=2, step=1)
    g["sps"]  = st.number_input("muestras por símbolo (sps)", value=int(g["sps"]), min_value=2, step=1)
    g["Fs"]   = st.number_input("Fs [Hz]", value=float(g["Fs"]), min_value=1e6, step=1e6, format="%.0f")
    g["Nsym"] = st.number_input("N símbolos", value=int(g["Nsym"]), min_value=64, step=64)
    g["Ptx"]  = st.number_input("Potencia Tx [W]", value=float(g["Ptx"]), min_value=1e-9, format="%.6f")
    g["lambda_nm"] = st.number_input("Longitud de onda [nm]", value=float(g.get("lambda_nm", 1550.0)), min_value=1200.0, max_value=1650.0, step=1.0)

with pcol:
    st.subheader("Pulso (Tx/Rx)")
    p = st.session_state["pulse"]
    p["type"] = st.selectbox("Tipo", ["RRC"], index=0)
    p["roll"] = st.number_input("roll-off", value=float(p["roll"]), min_value=0.01, max_value=1.0, step=0.01)
    p["span"] = st.number_input("span (símbolos)", value=int(p["span"]), min_value=1, step=1)

# ------------------------- cadena -------------------------

st.divider()
st.subheader("Cadena de bloques")

c1, c2, c3 = st.columns([1,1,2])
with c1:
    if st.button("➕ Añadir FIBER", use_container_width=True):
        blk = FiberBlock(type="fiber",
                         par=FiberPar(L=40e3, beta2=-2.1e-26, gamma=1.3e-3, dz=1.0, alpha=4.6e-5)).model_dump()
        ensure_uid(blk); st.session_state.chain.append(blk)
with c2:
    if st.button("➕ Añadir EDFA", use_container_width=True):
        blk = EdfaBlock(type="edfa", par=EdfaPar(G_dB=10.0, nsp=2.5)).model_dump()
        ensure_uid(blk); st.session_state.chain.append(blk)
with c3:
    cards_per_row = st.slider("Tarjetas por fila", min_value=2, max_value=6, value=4, help="Ajusta el zoom de la grilla")

columns_grid = st.columns(cards_per_row)
for i, blk in enumerate(st.session_state.chain):
    ensure_uid(blk)
    col = columns_grid[i % cards_per_row]
    with col:
        title = "FIBER" if blk["type"] == "fiber" else "EDFA"
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
                f'</div>', unsafe_allow_html=True
            )
        with bcol:
            xb1, xb2, xb3, xb4, xb5 = st.columns(5)
            with xb1:
                st.button("▲", key=f"up_{blk['_uid']}", on_click=move_block, args=(i,-1), help="Mover arriba", use_container_width=True)
            with xb2:
                st.button("▼", key=f"down_{blk['_uid']}", on_click=move_block, args=(i,+1), help="Mover abajo", use_container_width=True)
            with xb3:
                st.button("⧉", key=f"dup_{blk['_uid']}", on_click=duplicate_block, args=(i,), help="Duplicar", use_container_width=True)
            with xb4:
                st.button("✖", key=f"del_{blk['_uid']}", on_click=delete_block, args=(i,), help="Eliminar", use_container_width=True)
            with xb5:
                def _toggle_edit(k=i):
                    st.session_state.edit_idx = (None if st.session_state.edit_idx == k else k)
                st.button("⚙", key=f"edit_{blk['_uid']}", on_click=_toggle_edit, help="Editar", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

total_m = sum(b["par"]["L"] for b in st.session_state.chain if b["type"] == "fiber")
st.caption(f"Longitud total: {total_m/1e3:.1f} km")

# panel de edición
ei = st.session_state.edit_idx
if ei is not None and 0 <= ei < len(st.session_state.chain):
    blk = st.session_state.chain[ei]
    st.divider()
    st.subheader(f"Editar bloque #{ei+1} - {'FIBER' if blk['type']=='fiber' else 'EDFA'}")

    move_to = st.number_input("Mover a posición", min_value=1, max_value=len(st.session_state.chain), value=ei+1, step=1)
    cpos1, cpos2 = st.columns([0.2, 0.8])
    with cpos1:
        if st.button("Mover", use_container_width=True):
            move_block_to(ei, int(move_to))
            st.session_state.edit_idx = int(move_to) - 1
            st.rerun()

    st.markdown("### Parámetros")
    if blk["type"] == "fiber":
        par = blk["par"]
        par["L"]     = st.number_input("L [m]", value=float(par["L"]), min_value=1.0, step=100.0, key=f"L_{blk['_uid']}")
        par["beta2"] = st.number_input("β2 [s²/m]", value=float(par["beta2"]), step=1e-27, format="%.2e", key=f"b2_{blk['_uid']}")
        par["gamma"] = st.number_input("γ [1/(W·m)]", value=float(par["gamma"]), min_value=1e-6, step=1e-4, format="%.4f", key=f"gm_{blk['_uid']}")
        par["dz"]    = st.number_input("dz [m]", value=float(par["dz"]), min_value=0.1, step=0.1, key=f"dz_{blk['_uid']}")
        par["alpha"] = st.number_input("α [1/m]", value=float(par["alpha"]), min_value=0.0, step=1e-6, format="%.6e", key=f"al_{blk['_uid']}")
    else:
        par = blk["par"]
        par["G_dB"]  = st.number_input("Ganancia [dB]", value=float(par["G_dB"]), step=0.1, key=f"G_{blk['_uid']}")
        par["nsp"]   = st.number_input("nsp", value=float(par["nsp"]), min_value=0.5, step=0.1, key=f"nsp_{blk['_uid']}")

    if st.button("Cerrar edición", use_container_width=False):
        st.session_state.edit_idx = None
        st.rerun()

# ------------------------- ejecución -------------------------

st.divider()
st.subheader("Ejecución")

colA, colB, colC = st.columns(3)
with colA: gpu = st.toggle("Usar GPU (CuPy)", value=True)
with colB: dz_override = st.number_input("Override dz global [m] (opcional)", value=10.0, min_value=0.1, step=0.1)
with colC: step_const_km = st.number_input("Paso constelación [km]", value=5.0, min_value=0.5, step=0.5)

col1, col2, col3, col4 = st.columns(4)
with col1: insertion_db = st.number_input("Inserción [dB]", value=1.0, step=0.1)
with col2: splice_db    = st.number_input("Fusión [dB]", value=0.2, step=0.1)
with col3: do_const     = st.checkbox("Constelaciones", value=True)
with col4: do_eye       = st.checkbox("Eye final", value=True)

colh1, colh2 = st.columns(2)
with colh1: do_const3d_html = st.checkbox("3D interactivo (HTML)", value=True)
with colh2: const3d_html_pts = st.slider("Puntos por snapshot 3D", min_value=100, max_value=200, value=150, step=10)

plots_dir = st.text_input("Carpeta de plots", value="plots")
outdir    = st.text_input("Carpeta de logs", value="logs")

if st.button("▶ Ejecutar simulación", type="primary"):
    try:
        cfg_dict = {
            "global": st.session_state["global"],
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
            _execute(
                config=str(tmp),
                outdir=outdir,
                gpu=bool(gpu),
                dz=float(dz_override),
                insertion_db=float(insertion_db),
                use_insertion_loss=True,
                splice_db=float(splice_db),
                use_splice_loss=True,
                do_const=bool(do_const),
                step_const_km=float(step_const_km),
                do_eye=bool(do_eye),
                plots_dir=plots_dir,
                do_const3d=False, const3d_every=1, const3d_pts=1000,
                do_const3d_html=bool(do_const3d_html), const3d_html_pts=int(const3d_html_pts),
            )
            st.success("Simulación terminada")
        except Exception as e:
            st.error("Fallo durante la simulación")
            st.exception(e)

# ------------------------- helpers de resultados -------------------------

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
    # BER aprox por punto
    from math import erfc, sqrt
    Rb = float(g["Rb"]); M = int(g.get("M", 2))
    for pt in prof:
        osnr = pt.get("OSNR_dB", None)
        if osnr is None:
            pt["BER"] = None; continue
        OSNR_lin = 10.0**(osnr/10.0); SNR_lin = OSNR_lin * (Bo_Hz / max(Rb, 1.0))
        if M == 2:
            pt["BER"] = 0.5 * erfc(sqrt(max(SNR_lin,1e-12))/sqrt(2.0))
        elif M == 4:
            pt["BER"] = 0.5 * erfc(sqrt(0.5*max(SNR_lin,1e-12)))
        elif M == 16:
            pt["BER"] = (0.75/4.0) * erfc(sqrt(0.1*max(SNR_lin,1e-12)))
        else:
            pt["BER"] = 0.2 * erfc(sqrt(0.1*max(SNR_lin,1e-12)))
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

# ------------------------- resultados -------------------------

plots_p = pathlib.Path(plots_dir)
png1 = plots_p / "constelaciones.png"
png2 = plots_p / "potencia.png"
eye  = plots_p / "eye.png"

st.divider()
st.subheader("Resultados")

# resumen desde el último log
log = _read_last_log(outdir)
with st.expander("Resumen de la última ejecución", expanded=True):
    if log:
        res = log.get("result", {})
        cols = st.columns(5)
        cols[0].metric("Backend", res.get("backend", ""))
        cols[1].metric("Tiempo [s]", f"{res.get('elapsed_s', 0):.3f}")
        osnr = res.get("OSNR_final_dB", None)
        cols[2].metric("OSNR final [dB]", f"{osnr:.2f}" if osnr is not None else "n/a")
        pout = res.get("Pout_dBm", None)
        cols[3].metric("Pout [dBm]", f"{pout:.2f}" if pout is not None else "n/a")
        ber = res.get("BER_est_BPSK", None)
        cols[4].metric("BER BPSK", f"{ber:.3e}" if ber is not None else "n/a")
    else:
        st.info("Aún no hay logs en la carpeta seleccionada.")

# 3D interactivo generado por main
cands = sorted(plots_p.glob("constelaciones_3d*.html"), key=lambda p: p.stat().st_mtime, reverse=True)
if cands:
    html = cands[0].read_text(encoding="utf-8")
    components.html(html, height=800, scrolling=False)

# figuras existentes
cols = st.columns(3)
with cols[0]:
    if png1.exists(): st.image(str(png1), caption="Constelaciones (grid)")
with cols[1]:
    if png2.exists(): st.image(str(png2), caption="Evolución de potencia")
with cols[2]:
    if eye.exists():  st.image(str(eye), caption="Eye Diagram")

# ------------------------- perfiles z interactivos en GUI -------------------------

st.divider()
st.subheader("Perfiles z - analítico en GUI")

pc1, pc2, pc3, pc4 = st.columns(4)
with pc1: show_P = st.checkbox("Potencia [dBm]", value=True)
with pc2: show_O = st.checkbox("OSNR [dB]", value=True)
with pc3: show_B = st.checkbox("BER", value=True)
with pc4:
    Bo_GHz = st.number_input("Bo [GHz] para OSNR", value=12.5, min_value=0.1, max_value=100.0, step=0.1)

profile = build_profile_from_state(Bo_Hz=float(Bo_GHz)*1e9)

# plot interactivo con plotly si está disponible
try:
    import plotly.graph_objs as go
    traces = []
    z = [p["z_km"] for p in profile]
    if show_P:
        traces.append(go.Scatter(x=z, y=[p.get("P_dBm") for p in profile], mode="lines+markers", name="Potencia [dBm]"))
    if show_O:
        traces.append(go.Scatter(x=z, y=[p.get("OSNR_dB") for p in profile], mode="lines+markers", name="OSNR [dB]"))
    sec_axis = show_B
    layout = go.Layout(
        xaxis=dict(title="Distancia [km]"),
        yaxis=dict(title="dB / dBm"),
        yaxis2=dict(title="BER", overlaying="y", side="right", type="log") if sec_axis else None,
        legend=dict(orientation="h"),
        template="plotly_white",
        title="Perfiles z",
    )
    fig = go.Figure(data=traces, layout=layout)
    if show_B:
        fig.add_trace(go.Scatter(x=z, y=[p.get("BER") for p in profile], mode="lines+markers", name="BER", yaxis="y2"))
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
except Exception:
    # fallback simple
    st.line_chart({"z_km": [p["z_km"] for p in profile],
                   "P_dBm": [p.get("P_dBm") for p in profile]})

# descargas del perfil
col_dl1, col_dl2 = st.columns(2)
with col_dl1:
    st.download_button("⬇️ Descargar perfil CSV", data=_profile_to_csv_bytes(profile), file_name="perfil.csv", mime="text/csv")
with col_dl2:
    st.download_button("⬇️ Descargar perfil JSON", data=json.dumps(profile, indent=2).encode("utf-8"),
                       file_name="perfil.json", mime="application/json")
