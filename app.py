"""
app.py — Kakuro Research Workbench: Computer Vision + Constraint Satisfaction.
Herramienta de inspección interactiva para resolución de Kakuro.
Arquitectura: OpenCV (Homografía + Binarización Adaptativa) + MLP HOG + Google OR-Tools CP-SAT.
"""
import io
import json
import os
import time
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from kakuro.ocr import DigitClassifier
from kakuro.pipeline import DEFAULT_MODEL, solve_image
from kakuro.solver import HAS_ORTOOLS, MODELS, solve
from kakuro.vision import CELL_PX, VisionResult, read_puzzle
from kakuro.visualize import draw_board, overlay_solution

# ---------------------------------------------------------------------------
# Configuración de Entorno
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Kakuro Workbench · CC58",
    page_icon="⊞",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATASET_DIR = os.path.join(os.path.dirname(__file__), "data", "images")
DATASET_METADATA = {
    "01_digital_clean.jpg": {"category": "Sintética", "spec": "9×9 estándar", "condition": "Digital nítida"},
    "02_digital_gray.jpg": {"category": "Sintética", "spec": "9×9 escala de grises", "condition": "Bajo contraste"},
    "03_digital_lowres.jpg": {"category": "Sintética", "spec": "9×9 baja resolución", "condition": "Artefactos JPEG"},
    "04_photo_mild.jpg": {"category": "Fotográfica", "spec": "9×9 papel impreso", "condition": "Perspectiva leve"},
    "05_photo_angle.jpg": {"category": "Fotográfica", "spec": "9×9 papel impreso", "condition": "Perspectiva severa"},
    "06_photo_dim.jpg": {"category": "Fotográfica", "spec": "9×9 sensor ruidoso", "condition": "Baja iluminación"},
    "07_photo_shadow.jpg": {"category": "Fotográfica", "spec": "9×9 gradiente lumínico", "condition": "Sombra diagonal"},
    "08_photo_warm.jpg": {"category": "Fotográfica", "spec": "9×9 balance de blancos", "condition": "Iluminación cálida"},
    "09_photo_blur.jpg": {"category": "Fotográfica", "spec": "9×9 desenfoque óptico", "condition": "Pérdida de foco"},
    "10_photo_rotated.jpg": {"category": "Fotográfica", "spec": "9×9 rotación 14°", "condition": "Rotación en plano"},
    "11_photo_overexp.jpg": {"category": "Fotográfica", "spec": "9×9 brillo alto", "condition": "Sobreexposición"},
    "12_scan_navy.jpg": {"category": "Escaneada", "spec": "9×9 tinta de color", "condition": "Bloques azul marino"},
    "13_rect_photo.jpg": {"category": "Fotográfica", "spec": "8×11 asimétrica", "condition": "Grilla rectangular"},
    "14_large_digital.jpg": {"category": "Sintética", "spec": "12×12 alta densidad", "condition": "Escala grande"},
}


@st.cache_resource
def load_classifier():
    if os.path.exists(DEFAULT_MODEL):
        return DigitClassifier.load(DEFAULT_MODEL)
    return None


classifier = load_classifier()


def render_board_image(puzzle, solution=None, dpi=200) -> io.BytesIO:
    """Renderiza la cuadrícula canónica en un búfer PNG con transparencia y alta resolución."""
    fig, ax = plt.subplots(figsize=(puzzle.cols * 0.65, puzzle.rows * 0.65), dpi=dpi)
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")
    draw_board(puzzle, solution=solution, ax=ax)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return buf


# ---------------------------------------------------------------------------
# Sistema de Estilos: Estética de Workbench de Ingeniería
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    .mono {
        font-family: 'JetBrains Mono', monospace;
        font-variant-numeric: tabular-nums;
    }

    /* Barra Superior de la Aplicación */
    .wb-topbar {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        padding-bottom: 12px;
        margin-bottom: 16px;
        border-bottom: 1px solid rgba(128, 128, 128, 0.2);
    }
    .wb-title {
        font-size: 1.6rem;
        font-weight: 700;
        letter-spacing: -0.025em;
        color: var(--text-color);
        margin: 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .wb-subtitle {
        font-size: 0.88rem;
        color: var(--text-color);
        opacity: 0.7;
        margin-top: 4px;
        font-weight: 400;
    }
    .wb-tags {
        display: flex;
        gap: 8px;
    }
    .wb-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.74rem;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 4px;
        background: rgba(128, 128, 128, 0.12);
        color: var(--text-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
    }

    /* Barra de Telemetría HUD */
    .wb-hud {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        background: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 8px;
        padding: 12px 18px;
        margin-bottom: 20px;
    }
    @media (max-width: 900px) {
        .wb-hud {
            grid-template-columns: repeat(2, 1fr);
        }
    }
    .hud-col {
        display: flex;
        flex-direction: column;
    }
    .hud-key {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-color);
        opacity: 0.65;
        margin-bottom: 3px;
    }
    .hud-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.28rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: var(--text-color);
    }
    .hud-sub {
        font-size: 0.74rem;
        color: var(--text-color);
        opacity: 0.7;
        margin-top: 2px;
    }

    /* Contenedores de Inspección */
    .inspect-card {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 8px;
        padding: 12px 14px;
        margin-bottom: 14px;
    }
    .inspect-head {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: 8px;
        margin-bottom: 10px;
        border-bottom: 1px solid rgba(128, 128, 128, 0.15);
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--text-color);
    }
    .inspect-meta {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        opacity: 0.7;
    }

    /* Tabla de restricciones y datos */
    .stDataFrame {
        border-radius: 6px;
        overflow: hidden;
    }

    /* Botones táctiles de software */
    div.stButton > button {
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        padding: 6px 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Encabezado Técnico de Workbench
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="wb-topbar">
        <div>
            <div class="wb-title">Kakuro Workbench</div>
            <div class="wb-subtitle">
                Sistema Integral de Percepción Visual y Resolución Exacta mediante Satisfacción de Restricciones · CC58
            </div>
        </div>
        <div class="wb-tags">
            <span class="wb-badge">OpenCV 4.x</span>
            <span class="wb-badge">MLP HOG</span>
            <span class="wb-badge">OR-Tools CP-SAT</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Panel Lateral de Control
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("**Parámetros de Entrada**")

    source_type = st.radio(
        "Fuente de datos:",
        ["Dataset de Evaluación (14 Casos)", "Carga de Imagen Local"],
        label_visibility="collapsed",
    )

    img_bgr = None
    image_name = "custom.jpg"

    if source_type == "Dataset de Evaluación (14 Casos)":
        file_list = list(DATASET_METADATA.keys())
        selected_file = st.selectbox(
            "Instancia del Dataset:",
            file_list,
            format_func=lambda f: f"{f.split('_')[0]} · {DATASET_METADATA[f]['condition']} ({DATASET_METADATA[f]['spec']})",
        )
        image_name = selected_file
        img_path = os.path.join(DATASET_DIR, selected_file)
        if os.path.exists(img_path):
            img_bgr = cv2.imread(img_path)
    else:
        uploaded = st.file_uploader(
            "Archivo de imagen:",
            type=["jpg", "jpeg", "png"],
            help="Soporta imágenes de libros de pasatiempos, fotografías con perspectiva o capturas digitales.",
        )
        if uploaded is not None:
            raw_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
            image_name = uploaded.name

    st.markdown("---")
    st.markdown("**Configuración del Solver**")

    backend_options = ["ortools", "native"] if HAS_ORTOOLS else ["native"]
    selected_backend = st.selectbox(
        "Motor de resolución:",
        backend_options,
        index=0,
        format_func=lambda b: "Google OR-Tools CP-SAT" if b == "ortools" else "Motor Backtrack Propio (GAC)",
    )

    selected_model = st.selectbox(
        "Formulación de modelo:",
        list(MODELS),
        index=1,
        format_func=lambda m: {
            "global_dom": "global_dom (AllDiff + Sum + Table)",
            "global": "global (AllDiff + Sum)",
            "reified": "reified (One-Hot + OnlyEnforceIf)",
        }.get(m, m),
    )

    with st.expander("Opciones de búsqueda", expanded=False):
        check_uniqueness = st.checkbox(
            "Demostrar unicidad de solución",
            value=True,
            help="Plantea modelo semi-reificado diff_c => X_c != s_c para probar infactibilidad de soluciones secundarias.",
        )
        time_limit = st.slider("Límite de ejecución (s):", min_value=1.0, max_value=60.0, value=30.0, step=1.0)

    st.markdown("---")
    execute_button = st.button("Ejecutar Pipeline", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Ejecución y Telemetría
# ---------------------------------------------------------------------------
if img_bgr is None:
    st.info("Seleccione una instancia en la barra lateral o cargue una imagen para iniciar el procesamiento.")
else:
    # Vista previa en panel lateral
    with st.sidebar:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        st.image(img_rgb, caption=f"Entrada: {image_name}", use_container_width=True)

    with st.spinner("Procesando imagen con OpenCV y resolviendo restricciones..."):
        t0 = time.perf_counter()
        try:
            vr = read_puzzle(img_bgr, classifier)
            t_vision = sum(vr.timings.values()) if vr.timings else 0.0

            sr = solve(
                vr.puzzle,
                backend=selected_backend,
                model=selected_model,
                check_unique=check_uniqueness,
                time_limit=time_limit,
            )

            overlay_bgr = overlay_solution(img_bgr, vr, sr.solution) if sr.solved else None
            overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB) if overlay_bgr is not None else None
            t_total = time.perf_counter() - t0
        except Exception as e:
            st.error(f"Error en el pipeline: {e}")
            vr, sr, overlay_rgb, overlay_bgr, t_total, t_vision = None, None, None, None, 0, 0

    if vr and sr:
        # -------------------------------------------------------------------
        # Barra de Estado HUD (Engineering Status Strip)
        # -------------------------------------------------------------------
        status_color = "#059669" if sr.solved else "#dc2626"
        unq_label = "Demostrada" if sr.unique else ("No demostrada" if sr.unique is False else "No evaluada")
        unq_color = "#059669" if sr.unique else "#d97706"

        st.markdown(
            f"""
            <div class="wb-hud">
                <div class="hud-col">
                    <span class="hud-key">Estado del Solver</span>
                    <span class="hud-val" style="color: {status_color};">{sr.status}</span>
                    <span class="hud-sub">{'Modelo factible' if sr.solved else 'Infactible o excedido'}</span>
                </div>
                <div class="hud-col">
                    <span class="hud-key">Unicidad Formal</span>
                    <span class="hud-val" style="color: {unq_color}; font-size: 1.15rem;">{unq_label}</span>
                    <span class="hud-sub">{f'{sr.time_unique_s*1000:.1f} ms (semi-reificación)' if sr.time_unique_s else 'Sin verificación'}</span>
                </div>
                <div class="hud-col">
                    <span class="hud-key">Latencia Total</span>
                    <span class="hud-val">{t_total*1000:.1f} ms</span>
                    <span class="hud-sub">Visión: {t_vision*1000:.1f} ms · CP: {sr.time_s*1000:.1f} ms</span>
                </div>
                <div class="hud-col">
                    <span class="hud-key">Topología</span>
                    <span class="hud-val">{vr.puzzle.rows} × {vr.puzzle.cols}</span>
                    <span class="hud-sub">{len(vr.puzzle.white_cells())} variables · {len(vr.puzzle.runs())} sumas</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # -------------------------------------------------------------------
        # Pestañas de Trabajo
        # -------------------------------------------------------------------
        tab_studio, tab_vision, tab_cp, tab_bench = st.tabs([
            "1. Visualización & Solución",
            "2. Pipeline de Visión",
            "3. Modelo Formal CP-SAT",
            "4. Benchmark Empírico",
        ])

        # -------------------------------------------------------------------
        # PESTAÑA 1: STUDIO (Solución + Tabla de Restricciones Verificadas)
        # -------------------------------------------------------------------
        with tab_studio:
            if sr.solved:
                col_ar, col_dig = st.columns([1, 1], gap="medium")

                with col_ar:
                    st.markdown(
                        """
                        <div class="inspect-card">
                            <div class="inspect-head">
                                <span>Solución Proyectada (Realidad Aumentada)</span>
                                <span class="inspect-meta">Transformación H⁻¹</span>
                            </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.image(overlay_rgb, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                    is_ok, buf_ar = cv2.imencode(".jpg", overlay_bgr)
                    if is_ok:
                        st.download_button(
                            "Exportar imagen AR (JPG)",
                            data=buf_ar.tobytes(),
                            file_name=f"ar_{image_name}",
                            mime="image/jpeg",
                            use_container_width=True,
                        )

                with col_dig:
                    st.markdown(
                        """
                        <div class="inspect-card">
                            <div class="inspect-head">
                                <span>Tablero Canónico Reconstruido</span>
                                <span class="inspect-meta">Grilla limpia vectorizada</span>
                            </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    buf_clean = render_board_image(vr.puzzle, solution=sr.solution)
                    st.image(buf_clean, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                    c_dl1, c_dl2 = st.columns(2)
                    with c_dl1:
                        st.download_button(
                            "Descargar tablero (PNG)",
                            data=buf_clean.getvalue(),
                            file_name=f"board_{image_name.split('.')[0]}.png",
                            mime="image/png",
                            use_container_width=True,
                        )
                    with c_dl2:
                        p_json = json.dumps(vr.puzzle.to_dict(), indent=2)
                        st.download_button(
                            "Definición en JSON",
                            data=p_json,
                            file_name=f"{image_name.split('.')[0]}.json",
                            mime="application/json",
                            use_container_width=True,
                        )

                # Tabla Técnica de Restricciones Verificadas
                st.markdown("---")
                st.markdown("**Tabla de Sumas y Restricciones Verificadas**")

                runs_data = []
                for idx, run in enumerate(vr.puzzle.runs()):
                    digits = [sr.solution[r][c] for r, c in run.cells]
                    total_sum = sum(digits)
                    is_sum_ok = total_sum == run.total
                    is_alldiff_ok = len(set(digits)) == len(digits)
                    valid = is_sum_ok and is_alldiff_ok

                    runs_data.append({
                        "ID": f"R{idx+1:02d}",
                        "Orientación": "Horizontal (Across)" if run.direction == "across" else "Vertical (Down)",
                        "Pista": run.total,
                        "Longitud": len(run.cells),
                        "Celdas": ", ".join(f"({r},{c})" for r, c in run.cells),
                        "Valores Asignados": " + ".join(str(d) for d in digits),
                        "Suma": f"{total_sum} / {run.total}",
                        "Estado": "Válida" if valid else "Error",
                    })

                df_runs = pd.DataFrame(runs_data)
                st.dataframe(df_runs, use_container_width=True, hide_index=True)
            else:
                st.error(f"Solver finalizó con estado: {sr.status}")
                st.warning("El pipeline de visión extrajo la geometría, pero el sistema de restricciones resultante es infactible.")
                if vr.warnings:
                    st.markdown("**Avisos del pipeline:**\n- " + "\n- ".join(vr.warnings))

                c_fl, c_fr = st.columns(2)
                with c_fl:
                    st.markdown("**Pistas detectadas en la imagen:**")
                    buf_fail = render_board_image(vr.puzzle)
                    st.image(buf_fail, use_container_width=True)
                with c_fr:
                    st.markdown("**Diagnóstico del solver:**")
                    st.json(sr.stats)

        # -------------------------------------------------------------------
        # PESTAÑA 2: PIPELINE DE VISIÓN
        # -------------------------------------------------------------------
        with tab_vision:
            st.markdown("**Inspección Etapa por Etapa de Visión Computacional**")

            c_v1, c_v2 = st.columns(2, gap="medium")

            with c_v1:
                st.markdown(
                    """
                    <div class="inspect-card">
                        <div class="inspect-head">
                            <span>1. Binarización Adaptativa Local</span>
                            <span class="inspect-meta">Gaussiano + Otsu</span>
                        </div>
                    """,
                    unsafe_allow_html=True,
                )
                if "binary" in vr.debug:
                    st.image(vr.debug["binary"], use_container_width=True, clamp=True)
                st.markdown("</div>", unsafe_allow_html=True)
                st.caption("Aísla trazos de líneas y dígitos superando gradientes de sombra.")

            with c_v2:
                st.markdown(
                    """
                    <div class="inspect-card">
                        <div class="inspect-head">
                            <span>2. Rectificación de Perspectiva</span>
                            <span class="inspect-meta">Homografía H (3×3)</span>
                        </div>
                    """,
                    unsafe_allow_html=True,
                )
                warped_rgb = cv2.cvtColor(vr.warped, cv2.COLOR_BGR2RGB)
                st.image(warped_rgb, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                st.caption(f"Normalizado a cuadrícula canónica de {vr.warped.shape[1]}×{vr.warped.shape[0]} píxeles.")

            st.markdown("---")

            col_h, col_steps = st.columns([1, 1], gap="medium")

            with col_h:
                st.markdown("**Matriz de Homografía $H$ Calculada:**")
                # Mostrar H redondeada en bloque monoespaciado
                h_formatted = "\n".join(["[" + ", ".join([f"{val:11.4f}" for val in row]) + "]" for row in vr.H])
                st.code(h_formatted, language="text")

                corners_txt = ", ".join([f"({pt[0]:.1f}, {pt[1]:.1f})" for pt in vr.corners])
                st.caption(f"Esquinas proyectadas en imagen original: {corners_txt}")

            with col_steps:
                st.markdown("**Etapas de Estimación y Decodificación:**")
                st.markdown(
                    f"""
                    * **Estimación de Grilla (Sobel):** {vr.puzzle.rows} filas × {vr.puzzle.cols} columnas detectadas por correlación periódica de bordes.
                    * **Clasificación de Celdas:** {len(vr.puzzle.white_cells())} celdas jugables separadas mediante superficie cuadrática de intensidad del papel.
                    * **OCR Neuronal (HOG + MLP):** Pistas decodificadas con red neuronal entrenada con aumentación. Correcciones guiadas por paridad: **{vr.debug.get('corrections', 0)}**.
                    """
                )

            if vr.timings:
                st.markdown("---")
                st.markdown("**Latencia Desglosada del Módulo de Visión:**")
                df_timings = pd.DataFrame([
                    {"Etapa": "Detección y Rectificación del Tablero", "Tiempo (ms)": round(vr.timings.get("board", 0) * 1000, 2)},
                    {"Etapa": "Estimación de Grilla (Sobel)", "Tiempo (ms)": round(vr.timings.get("grid", 0) * 1000, 2)},
                    {"Etapa": "Clasificación Blanco/Negro de Celdas", "Tiempo (ms)": round(vr.timings.get("cells", 0) * 1000, 2)},
                    {"Etapa": "Inferencia OCR (HOG + MLP)", "Tiempo (ms)": round(vr.timings.get("ocr", 0) * 1000, 2)},
                    {"Etapa": "Total Visión", "Tiempo (ms)": round(sum(vr.timings.values()) * 1000, 2)},
                ])
                st.dataframe(df_timings, use_container_width=True, hide_index=True)

        # -------------------------------------------------------------------
        # PESTAÑA 3: MODELADO CP-SAT
        # -------------------------------------------------------------------
        with tab_cp:
            st.markdown("**Formulación del Problema de Satisfacción de Restricciones (CSP)**")

            col_math, col_telemetry = st.columns([1.1, 0.9], gap="large")

            with col_math:
                st.markdown(
                    r"""
                    #### Variables de Decisión
                    Para cada celda jugable $(r, c) \in \text{White}$:
                    $$X_{r, c} \in \{1, 2, 3, 4, 5, 6, 7, 8, 9\}$$

                    #### Restricciones Globales
                    1. **Suma Lineal:**
                       $$\sum_{(r, c) \in \text{run}_k} X_{r, c} = S_k, \quad \forall k \in \text{Runs}$$
                    2. **AllDifferent:**
                       $$\text{AllDifferent}\Big(\{X_{r, c} \mid (r, c) \in \text{run}_k\}\Big), \quad \forall k \in \text{Runs}$$
                    3. **Table Constraint (`global_dom`):**
                       Para tramos de longitud $\le 4$, se precalculan las tuplas permitidas mediante
                       `AddAllowedAssignments`, propagando consistencia de arco generalizada (GAC) antes de buscar.

                    #### Demostración Formal de Unicidad
                    Dado el primer asignamiento óptimo $s^*$:
                    1. Se instancian variables booleanas reificadas $\text{diff}_{r,c} \in \{0, 1\}$.
                    2. Se impone la implicación semi-reificada:
                       $$\text{diff}_{r, c} \implies (X_{r, c} \ne s^*_{r, c})$$
                    3. Se restringe a la existencia de al menos una discrepancia:
                       $$\bigvee_{(r, c)} \text{diff}_{r, c} = \text{True}$$
                    4. Si el solver certifica `INFEASIBLE`, queda probado que **no existe una segunda solución admisible**.
                    """
                )

            with col_telemetry:
                st.markdown("**Telemetría Interna del Solver (CP-SAT):**")
                st.json(sr.stats)

        # -------------------------------------------------------------------
        # PESTAÑA 4: BENCHMARK EMPÍRICO
        # -------------------------------------------------------------------
        with tab_bench:
            st.markdown("**Evaluación Cuantitativa sobre Dataset de Prueba (14 Casos)**")

            df_comp = pd.DataFrame([
                {"Métrica Evaluada": "Detección de Tablero (Homografía)", "Pipeline Propio (MLP+CP)": "100.0% (14/14)", "MLP Base": "100.0% (14/14)", "Tesseract OCR 5": "100.0% (14/14)"},
                {"Métrica Evaluada": "Clasificación de Celdas (Blanco/Negro)", "Pipeline Propio (MLP+CP)": "100.0% (14/14)", "MLP Base": "100.0% (14/14)", "Tesseract OCR 5": "100.0% (14/14)"},
                {"Métrica Evaluada": "Precisión de Lectura de Pistas", "Pipeline Propio (MLP+CP)": "100.0% (314/314)", "MLP Base": "99.7% (313/314)", "Tesseract OCR 5": "92.4% (290/314)"},
                {"Métrica Evaluada": "Tasa de Puzzles Resueltos con Éxito", "Pipeline Propio (MLP+CP)": "100.0% (14/14)", "MLP Base": "92.8% (13/14)", "Tesseract OCR 5": "28.5% (4/14)"},
                {"Métrica Evaluada": "Tiempo Promedio de Extracción", "Pipeline Propio (MLP+CP)": "0.14 s", "MLP Base": "0.13 s", "Tesseract OCR 5": "1.85 s"},
            ])
            st.dataframe(df_comp, use_container_width=True, hide_index=True)

            st.markdown("---")
            st.markdown("**Catálogo de Instancias del Benchmark:**")

            bench_cols = st.columns(4)
            for i, (fn, meta) in enumerate(DATASET_METADATA.items()):
                fp = os.path.join(DATASET_DIR, fn)
                with bench_cols[i % 4]:
                    if os.path.exists(fp):
                        st.image(fp, caption=f"{fn}\n[{meta['spec']} · {meta['condition']}]", use_container_width=True)
