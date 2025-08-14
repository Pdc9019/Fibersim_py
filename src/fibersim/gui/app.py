# src/fibersim/gui/app.py
from __future__ import annotations
import json, pathlib, uuid
from typing import Any, Dict, List
import streamlit as st
import streamlit.components.v1 as components

# Modelos / ejecución del simulador
from fibersim.schema import SimConfig, FiberBlock, EdfaBlock, FiberPar, EdfaPar
from fibersim.main import _execute


# ------------------------- utilidades -------------------------

def ensure_uid(blk: dict) -> None:
    """Asegura un identificador estable por tarjeta."""
    if "_uid" not in blk:
        blk["_uid"] = uuid.uuid4().hex

def summarize_block(blk: dict) -> str:
    """Resumen MUY corto para la mini-tarjeta (compacto)."""
    t = blk["type"]
    if t == "fiber":
        p = blk["par"]
        L_km = p["L"] / 1e3
        alpha_db_km = 4.343 * p["alpha"] * 1e3  # potencia dB/km
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
    """Mueve el bloque en idx_from a la posición idx_to (1-based)."""
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


# ------------------------- página / estado -------------------------

st.set_page_config(page_title="FiberSim GUI", layout="wide")

# === Estilos compactos, grid de mini-tarjetas y chips por tipo ===
st.markdown("""
<style>
div.block-container { padding-top: 1.0rem; }

/* Mini-tarjeta */
.mini-card {
  border: 1px solid rgba(255,255,255,.12);
  background: rgba(255,255,255,.03);
  border-radius: 10px;
  padding: 8px 10px;
  margin: 6px 0;
}
.mini-card:hover { background: rgba(255,255,255,.05); }

/* Cabecera compacta */
.mini-head { display:flex; align-items:center; gap:.45rem; white-space:nowrap; overflow:hidden; }
.mini-idx { font-weight:700; opacity:.85; }
.mini-sub { opacity:.82; font-size:.85rem; overflow:hidden; text-overflow:ellipsis; }

/* Chips por tipo */
.chip {
  font-size:.78rem; padding:.16rem .45rem; border-radius:8px; font-weight:700;
}
.chip.fiber { background:#1e3a8a22; color:#93c5fd; border:1px solid #1e3a8a44; }
.chip.edfa  { background:#14532d22; color:#86efac; border:1px solid #14532d44; }

/* Botones pequeños y columnas más juntas */
div[data-testid="stHorizontalBlock"] { gap:.25rem; }
</style>
""", unsafe_allow_html=True)

# Estado inicial
if "chain" not in st.session_state:
    st.session_state.chain: List[dict] = []
if "global" not in st.session_state:
    st.session_state["global"] = dict(Rb=32e9, M=2, sps=32, Fs=1.024e12, Nsym=4096, Ptx=1e-2)
if "pulse" not in st.session_state:
    st.session_state["pulse"] = dict(type="RRC", roll=0.1, span=10)
if "edit_idx" not in st.session_state:
    st.session_state.edit_idx: int | None = None


# ------------------------- carga / guardado -------------------------

st.sidebar.header("⚙️ Configuración")
upl = st.sidebar.file_uploader("Cargar JSON", type=["json"])
if upl:
    try:
        cfg = SimConfig.model_validate(json.loads(upl.read().decode("utf-8")))
        st.session_state["global"] = cfg.global_.model_dump()
        st.session_state["pulse"]  = cfg.pulse.model_dump()
        st.session_state.chain     = [b.model_dump() for b in cfg.chain]
        for b in st.session_state.chain: ensure_uid(b)
        st.sidebar.success("Configuración cargada ✅")
    except Exception as e:
        st.sidebar.error(f"Error al cargar: {e}")

def export_json() -> str:
    data = {"global": st.session_state["global"], "pulse": st.session_state["pulse"], "chain": st.session_state.chain}
    return json.dumps(data, indent=2)

st.sidebar.download_button("💾 Descargar JSON", data=export_json(), file_name="config.json", mime="application/json")


# ------------------------- parámetros globales/pulso -------------------------

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
    p["type"] = st.selectbox("Tipo", ["RRC"], index=0)
    p["roll"] = st.number_input("roll-off", value=float(p["roll"]), min_value=0.01, max_value=1.0, step=0.01)
    p["span"] = st.number_input("span (símbolos)", value=int(p["span"]), min_value=1, step=1)


# ------------------------- cadena (builder) -------------------------

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
    cards_per_row = st.slider("Tarjetas por fila", min_value=2, max_value=6, value=4,
                              help="Ajusta el zoom de la grilla")

# ========== GRID DE MINI-TARJETAS ==========
columns_grid = st.columns(cards_per_row)
for i, blk in enumerate(st.session_state.chain):
    ensure_uid(blk)
    col = columns_grid[i % cards_per_row]
    with col:
        title = "FIBER" if blk["type"] == "fiber" else "EDFA"
        badge_cls = "fiber" if blk["type"] == "fiber" else "edfa"
        subtitle = summarize_block(blk)

        st.markdown('<div class="mini-card">', unsafe_allow_html=True)

        # cabecera (texto compacto)
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
        # botones: ▲ ▼ ⧉ ✖ ⚙
        with bcol:
            xb1, xb2, xb3, xb4, xb5 = st.columns(5)
            with xb1:
                st.button("▲", key=f"up{i}", on_click=move_block, args=(i,-1),
                          help="Mover arriba", use_container_width=True)
            with xb2:
                st.button("▼", key=f"down{i}", on_click=move_block, args=(i,+1),
                          help="Mover abajo", use_container_width=True)
            with xb3:
                st.button("⧉", key=f"dup{i}", on_click=duplicate_block, args=(i,),
                          help="Duplicar", use_container_width=True)
            with xb4:
                st.button("✖", key=f"del{i}", on_click=delete_block, args=(i,),
                          help="Eliminar", use_container_width=True)
            with xb5:
                def _toggle_edit(k=i):
                    st.session_state.edit_idx = (None if st.session_state.edit_idx == k else k)
                st.button("⚙", key=f"edit{i}", on_click=_toggle_edit,
                          help="Editar", use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

# resumen longitud total
total_m = sum(b["par"]["L"] for b in st.session_state.chain if b["type"] == "fiber")
st.caption(f"Longitud total: {total_m/1e3:.1f} km")


# ========== PANEL DE EDICIÓN (único) ==========
ei = st.session_state.edit_idx
if ei is not None and 0 <= ei < len(st.session_state.chain):
    blk = st.session_state.chain[ei]
    st.divider()
    st.subheader(f"Editar bloque #{ei+1} — {'FIBER' if blk['type']=='fiber' else 'EDFA'}")

    # mover rápido a posición N
    move_to = st.number_input("Mover a posición", min_value=1, max_value=len(st.session_state.chain),
                              value=ei+1, step=1)
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
with col2: splice_db    = st.number_input("Fusión (splice) [dB]", value=0.2, step=0.1)
with col3: do_const     = st.checkbox("Constelaciones", value=True)
with col4: do_eye       = st.checkbox("Eye Diagram final", value=True)

colh1, colh2 = st.columns(2)
with colh1: do_const3d_html = st.checkbox("3D interactivo (HTML)", value=True)
with colh2: const3d_html_pts = st.slider("Puntos por snapshot 3D", min_value=200, max_value=4000, value=1200, step=200)

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
                # 3D: solo HTML desde GUI
                do_const3d=False, const3d_every=1, const3d_pts=1000,
                do_const3d_html=bool(do_const3d_html), const3d_html_pts=int(const3d_html_pts),
            )
            st.success("Simulación terminada ✅")
        except Exception as e:
            st.error("Fallo durante la simulación")
            st.exception(e)


# ------------------------- resultados -------------------------

plots_p = pathlib.Path(plots_dir)
png1 = plots_p / "constelaciones.png"
png2 = plots_p / "potencia.png"
eye  = plots_p / "eye.png"

st.divider()
st.subheader("Resultados")

# 3D interactivo (toma el HTML más reciente)
cands = sorted(plots_p.glob("constelaciones_3d*.html"),
               key=lambda p: p.stat().st_mtime, reverse=True)
if cands:
    html = cands[0].read_text(encoding="utf-8")
    components.html(html, height=800, scrolling=False)

cols = st.columns(3)
with cols[0]:
    if png1.exists(): st.image(str(png1), caption="Constelaciones (grid)")
with cols[1]:
    if png2.exists(): st.image(str(png2), caption="Evolución de potencia")
with cols[2]:
    if eye.exists():  st.image(str(eye), caption="Eye Diagram")
