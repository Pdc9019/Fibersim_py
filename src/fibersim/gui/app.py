from __future__ import annotations
import json, pathlib
from typing import Any, Dict
import streamlit as st
import streamlit.components.v1 as components
from streamlit_sortables import sort_items

from fibersim.schema import (
    SimConfig, FiberBlock, EdfaBlock,
    FiberPar, EdfaPar
)
from fibersim.main import _execute


# ---------- Utilidad: resumen compacto por bloque ----------
def summarize_block(blk: dict) -> str:
    """Título corto para mostrar en la lista y en el expander."""
    t = blk["type"]
    if t == "fiber":
        par = blk["par"]
        L_km = par["L"] / 1e3
        # alpha [1/m] -> dB/km (potencia): 4.343 * alpha * 1e3
        alpha_db_km = 4.343 * par["alpha"] * 1e3
        return (
            f"FIBER · L={L_km:.0f} km · β2={par['beta2']:.2e} s²/m · "
            f"γ={par['gamma']:.2e} 1/(W·m) · α≈{alpha_db_km:.2f} dB/km"
        )
    elif t == "edfa":
        par = blk["par"]
        return f"EDFA · G={par['G_dB']:.1f} dB · nsp={par['nsp']:.2f}"
    else:
        return t.upper()


st.set_page_config(page_title="FiberSim GUI", layout="wide")

# ---------- Estado inicial ----------
if "chain" not in st.session_state:
    st.session_state.chain = []  # lista de bloques (dicts)
if "global" not in st.session_state:
    st.session_state["global"] = dict(
        Rb=32e9, M=2, sps=32, Fs=1.024e12, Nsym=4096, Ptx=1e-2
    )
if "pulse" not in st.session_state:
    st.session_state["pulse"] = dict(type="RRC", roll=0.1, span=10)

# ---------- Sidebar: Cargar / Guardar ----------
st.sidebar.header("⚙️ Configuración")
uploaded = st.sidebar.file_uploader("Cargar JSON", type=["json"])
if uploaded is not None:
    try:
        cfg = json.loads(uploaded.read().decode("utf-8"))
        cfg_model = SimConfig.model_validate(cfg)  # valida y normaliza
        st.session_state["global"] = cfg_model.global_.model_dump()
        st.session_state["pulse"] = cfg_model.pulse.model_dump()
        st.session_state.chain = [b.model_dump() for b in cfg_model.chain]
        st.sidebar.success("Configuración cargada ✅")
    except Exception as e:
        st.sidebar.error(f"Error al cargar: {e}")

def export_json() -> str:
    data = {
        "global": st.session_state["global"],
        "pulse": st.session_state["pulse"],
        "chain": st.session_state.chain,
    }
    return json.dumps(data, indent=2)

st.sidebar.download_button(
    "💾 Descargar JSON", data=export_json(),
    file_name="config.json", mime="application/json"
)

# ---------- Edición de parámetros globales ----------
st.title("FiberSim — Construcción de enlace")

gcol, pcol = st.columns(2)
with gcol:
    st.subheader("Parámetros Globales")
    g = st.session_state["global"]
    g["Rb"]   = st.number_input("Rb [baud]", value=float(g["Rb"]), min_value=1e6, step=1e6, format="%.0f")
    g["M"]    = st.number_input("Orden M", value=int(g["M"]), min_value=2, step=1)
    g["sps"]  = st.number_input("muestras/símbolo (sps)", value=int(g["sps"]), min_value=2, step=1)
    g["Fs"]   = st.number_input("Fs [Hz]", value=float(g["Fs"]), min_value=1e6, step=1e6, format="%.0f")
    g["Nsym"] = st.number_input("N símbolos", value=int(g["Nsym"]), min_value=64, step=64)
    g["Ptx"]  = st.number_input("Potencia Tx [W]", value=float(g["Ptx"]), min_value=1e-6, format="%.6f")

with pcol:
    st.subheader("Pulso (Tx/Rx)")
    p = st.session_state["pulse"]
    p["type"] = st.selectbox("Tipo", options=["RRC"], index=0)
    p["roll"] = st.number_input("roll-off", value=float(p["roll"]), min_value=0.01, max_value=1.0, step=0.01)
    p["span"] = st.number_input("span (símbolos)", value=int(p["span"]), min_value=1, step=1)

st.divider()
st.subheader("Cadena de bloques")

# ---------- Añadir bloques ----------
add1, add2 = st.columns(2)
with add1:
    if st.button("➕ Añadir FIBER"):
        st.session_state.chain.append(
            FiberBlock(type="fiber", par=FiberPar(
                L=40e3, beta2=-2.1e-26, gamma=1.3e-3, dz=1.0, alpha=4.6e-5
            )).model_dump()
        )
with add2:
    if st.button("➕ Añadir EDFA"):
        st.session_state.chain.append(
            EdfaBlock(type="edfa", par=EdfaPar(G_dB=10.0, nsp=2.5)).model_dump()
        )

# ---------- Reordenar por drag-and-drop ----------
if st.session_state.chain:
    st.caption("Arrastra para reordenar la cadena:")
    labels = [f"[{i}] {summarize_block(b)}" for i, b in enumerate(st.session_state.chain)]
    sorted_labels = sort_items(labels)  # devuelve la lista en el nuevo orden
    if sorted_labels and sorted_labels != labels:
        new_order = [int(s.split(']')[0][1:]) for s in sorted_labels]
        st.session_state.chain = [st.session_state.chain[i] for i in new_order]
        st.rerun()

# ---------- Lista editable (expander por bloque) ----------
for idx, blk in enumerate(st.session_state.chain):
    with st.expander(f"{idx+1}. {summarize_block(blk)}", expanded=False):
        c1, c2, c3 = st.columns([1,1,1])
        with c1:
            if st.button("▲", key=f"up{idx}") and idx > 0:
                st.session_state.chain[idx-1], st.session_state.chain[idx] = \
                    st.session_state.chain[idx], st.session_state.chain[idx-1]
                st.rerun()
        with c2:
            if st.button("▼", key=f"down{idx}") and idx < len(st.session_state.chain)-1:
                st.session_state.chain[idx+1], st.session_state.chain[idx] = \
                    st.session_state.chain[idx], st.session_state.chain[idx+1]
                st.rerun()
        with c3:
            if st.button("🗑️ Borrar", key=f"del{idx}"):
                st.session_state.chain.pop(idx)
                st.rerun()

        # Edición de parámetros
        if blk["type"] == "fiber":
            par = blk["par"]
            par["L"]     = st.number_input("L [m]", value=float(par["L"]), min_value=1.0, step=100.0, key=f"L{idx}")
            par["beta2"] = st.number_input("beta2 [s^2/m]", value=float(par["beta2"]), step=1e-27, format="%.2e", key=f"b2{idx}")
            par["gamma"] = st.number_input("gamma [1/(W·m)]", value=float(par["gamma"]), min_value=1e-6, step=1e-4, format="%.4f", key=f"gm{idx}")
            par["dz"]    = st.number_input("dz [m]", value=float(par["dz"]), min_value=0.1, step=0.1, key=f"dz{idx}")
            par["alpha"] = st.number_input("alpha [1/m]", value=float(par["alpha"]), min_value=0.0, step=1e-6, format="%.6e", key=f"al{idx}")
        else:  # edfa
            par = blk["par"]
            par["G_dB"] = st.number_input("Ganancia [dB]", value=float(par["G_dB"]), step=0.1, key=f"G{idx}")
            par["nsp"]  = st.number_input("nsp", value=float(par["nsp"]), min_value=0.5, step=0.1, key=f"nsp{idx}")

# Vista previa de longitud total
total_m = sum(b["par"]["L"] for b in st.session_state.chain if b["type"] == "fiber")
st.caption(f"Longitud total: {total_m/1e3:.1f} km")

# ---------- Ejecución ----------
st.divider()
st.subheader("Ejecución")

colA, colB, colC = st.columns(3)
with colA:
    gpu = st.toggle("Usar GPU (CuPy)", value=True)
with colB:
    dz_override = st.number_input("Override dz global [m] (opcional)", value=10.0, min_value=0.1, step=0.1)
with colC:
    step_const_km = st.number_input("Paso constelación [km]", value=5.0, min_value=0.5, step=0.5)

col1, col2, col3, col4 = st.columns(4)
with col1:
    insertion_db = st.number_input("Inserción [dB]", value=1.0, step=0.1)
with col2:
    splice_db = st.number_input("Fusión (splice) [dB]", value=0.2, step=0.1)
with col3:
    do_const = st.checkbox("Constelaciones", value=True)
with col4:
    do_eye = st.checkbox("Eye Diagram final", value=True)

colh1, colh2 = st.columns(2)
with colh1:
    do_const3d_html = st.checkbox("3D interactivo (HTML)", value=True)
with colh2:
    const3d_html_pts = st.slider("Puntos por snapshot 3D", min_value=200, max_value=4000, value=1200, step=200)

plots_dir = st.text_input("Carpeta de plots", value="plots")
outdir = st.text_input("Carpeta de logs", value="logs")

# Ejecutar
if st.button("▶ Ejecutar simulación", type="primary"):
    try:
        cfg_dict = {
            "global": st.session_state["global"],
            "pulse": st.session_state["pulse"],
            "chain": st.session_state.chain,
        }
        cfg_model = SimConfig.model_validate(cfg_dict)
        cfg_norm = cfg_model.model_dump(by_alias=True)
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
                # 3D (HTML) desde la GUI
                do_const3d=False,
                const3d_every=1,
                const3d_pts=1000,
                do_const3d_html=bool(do_const3d_html),
                const3d_html_pts=int(const3d_html_pts),
            )
            st.success("Simulación terminada ✅")
        except Exception as e:
            st.error("Fallo durante la simulación")
            st.exception(e)

# ---------- Resultados ----------
plots_p = pathlib.Path(plots_dir)
png1 = plots_p / "constelaciones.png"
png2 = plots_p / "potencia.png"
eye  = plots_p / "eye.png"

st.divider()
st.subheader("Resultados")

cols = st.columns(3)
with cols[0]:
    if png1.exists():
        st.image(str(png1), caption="Constelaciones (grid)")
with cols[1]:
    if png2.exists():
        st.image(str(png2), caption="Evolución de potencia")
with cols[2]:
    if eye.exists():
        st.image(str(eye), caption="Eye Diagram")

# 3D interactivo: elegir el HTML más reciente y aplicar cache-busting
cands = sorted(
    plots_p.glob("constelaciones_3d*.html"),
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)
if cands:
    html3d = cands[0]
    html = html3d.read_text(encoding="utf-8")
    mtime = int(html3d.stat().st_mtime)
    # Cache-busting: cambia ligeramente el contenido para forzar rerender del iframe
    html += f"\n<!-- cache-bust:{mtime} -->"
    components.html(html, height=800, scrolling=False)
