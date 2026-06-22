"""
app.py – Streamlit front-end for the Suplex Simplex solver.

Run with:
    streamlit run src/interface/app.py
"""

from __future__ import annotations

import sys
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.backend.utils import parse_text, to_fpi, LP, format_tableau
from src.backend.suplex import solve, SolveResult
from src.backend.llm_parser import parse_natural_language, list_available_models, OLLAMA_MODEL

# ─────────────────────────── Page config ─────────────────────────────────────

_ICON_PATH = Path(__file__).resolve().parent.parent.parent / "doc" / "src" / "small_icon.png"
_PAGE_ICON = Image.open(_ICON_PATH)

st.set_page_config(
    page_title="Project Suplex – Simplex Optimization",
    page_icon=_PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────── Session state helpers ───────────────────────────

def _init_state():
    defaults = {
        "lp_text": "",
        "result": None,
        "step_idx": 0,
        "diary": [],
        "llm_rationale": "",
        "n_vars": 2,
        "n_cons": 2,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ─────────────────────────── Sidebar ─────────────────────────────────────────

with st.sidebar:
    st.image(str(_ICON_PATH), width="stretch")
    st.markdown(
        "<h2 style='text-align:center; margin-top:0.2rem; margin-bottom:0;'>Project Suplex</h2>"
        "<p style='text-align:center; color:gray; margin-top:0;'><em>Simplex Optimization</em></p>",
        unsafe_allow_html=True,
    )
    st.divider()
    st.title("⚙️ Configurações")
    policy = st.selectbox("Regra de entrada (pivot)", ["largest", "bland", "smallest"], index=0)
    decimals = st.slider("Casas decimais", 2, 8, 4)
    digits   = st.slider("Largura coluna (tableau)", 6, 16, 10)
    st.divider()
    st.markdown("**Suplex Bot** · Fase I + Fase II  \nSimplex totalmente manual (sem scipy).")
    st.markdown("---")
    llm_model = st.text_input("Modelo Ollama", value=OLLAMA_MODEL)

# ─────────────────────────── Tabs ────────────────────────────────────────────

tab_setup, tab_llm, tab_diary = st.tabs([
    "🔧 Setup & Visualização",
    "🤖 Llama 3 (NLP)",
    "📓 Diário / Logs",
])

# ══════════════════════════════════════════════════════════════════════════════
#  Helper functions (must be defined before the tab code that calls them)
# ══════════════════════════════════════════════════════════════════════════════

EXAMPLES_DIR = Path(__file__).resolve().parent.parent.parent / "examples"


def _example_text() -> str:
    return "2\n2\n1 1\n5 4\n6 1 <= 6\n1 1 <= 4"


def _list_examples() -> list[Path]:
    return sorted(EXAMPLES_DIR.glob("*.txt"))


def _build_feasible_region_plot(lp: LP, result: SolveResult, decimals: int) -> go.Figure:
    """Build a Plotly figure showing the feasible region and simplex path."""
    upper = max(float(np.max(np.abs(lp.b))) * 1.5, 10.0)
    xs = np.linspace(0, upper, 600)

    fig = go.Figure()

    grid_n = 300
    gx = np.linspace(0, upper, grid_n)
    gy = np.linspace(0, upper, grid_n)
    GX, GY = np.meshgrid(gx, gy)
    feasible = np.ones((grid_n, grid_n), dtype=bool)

    for i in range(lp.n_cons):
        a1, a2 = lp.A[i, 0], lp.A[i, 1]
        b = lp.b[i]
        lhs = a1 * GX + a2 * GY
        if lp.signs[i] == "<=":
            feasible &= lhs <= b + 1e-8
        elif lp.signs[i] == ">=":
            feasible &= lhs >= b - 1e-8

    if lp.domain[0] == 1:
        feasible &= GX >= -1e-8
    if lp.domain[1] == 1:
        feasible &= GY >= -1e-8

    fig.add_trace(go.Heatmap(
        x=gx, y=gy,
        z=feasible.astype(float),
        colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,120,200,0.18)"]],
        showscale=False,
        hoverinfo="skip",
    ))

    colors = ["#e74c3c", "#2ecc71", "#9b59b6", "#f39c12", "#1abc9c"]
    for i in range(lp.n_cons):
        a1, a2 = lp.A[i, 0], lp.A[i, 1]
        b = lp.b[i]
        if abs(a2) > 1e-9:
            ys = (b - a1 * xs) / a2
            mask = (ys >= -0.5) & (ys <= upper * 1.1)
            fig.add_trace(go.Scatter(
                x=xs[mask], y=ys[mask],
                mode="lines",
                name=f"R{i+1}: {a1}x₁+{a2}x₂{lp.signs[i]}{b}",
                line=dict(color=colors[i % len(colors)], width=2),
            ))
        elif abs(a1) > 1e-9:
            xv = b / a1
            fig.add_vline(x=xv, line_color=colors[i % len(colors)],
                          annotation_text=f"R{i+1}", annotation_position="top right")

    path_x, path_y = [], []
    for t_step, b_step in zip(result.tableau_history, result.basis_history):
        n_fpi = t_step.shape[1] - 1
        xp = np.zeros(n_fpi)
        for idx, bv in enumerate(b_step):
            if bv < n_fpi:
                xp[bv] = t_step[idx, -1]
        path_x.append(float(xp[0]) if len(xp) > 0 else 0.0)
        path_y.append(float(xp[1]) if len(xp) > 1 else 0.0)

    if len(path_x) > 1:
        fig.add_trace(go.Scatter(
            x=path_x, y=path_y,
            mode="lines+markers",
            name="Caminho Simplex",
            line=dict(color="#f39c12", width=2.5, dash="dashdot"),
            marker=dict(size=8, color="#f39c12", symbol="circle"),
        ))

    if result.x_primal is not None:
        ox, oy = float(result.x_primal[0]), float(result.x_primal[1])
        fig.add_trace(go.Scatter(
            x=[ox], y=[oy],
            mode="markers+text",
            name=f"Ótimo Z={result.z:.{decimals}f}",
            marker=dict(size=14, color="#e74c3c", symbol="star"),
            text=[f"Z={result.z:.{decimals}f}"],
            textposition="top right",
        ))

    fig.update_layout(
        xaxis_title="x₁", yaxis_title="x₂",
        xaxis=dict(range=[-0.5, upper]),
        yaxis=dict(range=[-0.5, upper]),
        legend=dict(bgcolor="rgba(0,0,0,0.05)"),
        plot_bgcolor="white",
        height=520,
    )
    return fig


def _render_economic_interpretation(result: SolveResult, decimals: int):
    """Render a markdown block with economic interpretation of the solution."""
    st.subheader("💡 Interpretação Econômica")
    lines = []
    lines.append("#### Solução Primal")
    lines.append(
        "As variáveis de decisão no ponto ótimo representam os níveis de produção/alocação:"
    )
    for j, name in enumerate(result.var_names):
        val = result.x_primal[j]
        if abs(val) > 1e-9:
            lines.append(f"- **{name}** = `{val:.{decimals}f}` → recurso/produto utilizado no nível ótimo.")
        else:
            lines.append(f"- **{name}** = `0` → não utilizado na solução ótima.")
    lines.append("\n#### Solução Dual (Preços-Sombra)")
    lines.append(
        "Cada preço-sombra **yᵢ** indica o ganho marginal em Z se o RHS da restrição i "
        "aumentar em 1 unidade:"
    )
    for i, yi in enumerate(result.y_dual):
        if abs(yi) > 1e-9:
            lines.append(
                f"- **y{i+1}** = `{yi:.{decimals}f}` → relaxar a restrição {i+1} em 1 unidade "
                f"aumenta Z em `{yi:.{decimals}f}`."
            )
        else:
            lines.append(f"- **y{i+1}** = `0` → restrição {i+1} não está ativa (folga).")
    st.markdown("\n".join(lines))


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 – Setup & Visualização
# ══════════════════════════════════════════════════════════════════════════════

with tab_setup:
    st.header("Setup do Problema")

    input_mode = st.radio("Modo de entrada", ["Manual (sliders)", "Texto estruturado"], horizontal=True)

    # ── Manual input ──────────────────────────────────────────────────────────
    if input_mode == "Manual (sliders)":
        col_nv, col_nc = st.columns(2)
        with col_nv:
            st.session_state.n_vars = st.number_input("Número de variáveis", 1, 10, st.session_state.n_vars, step=1)
        with col_nc:
            st.session_state.n_cons = st.number_input("Número de restrições", 1, 20, st.session_state.n_cons, step=1)

        nv = int(st.session_state.n_vars)
        nc = int(st.session_state.n_cons)

        st.subheader("Domínio das variáveis")
        domain_cols = st.columns(nv)
        domain_vals = []
        for j, col in enumerate(domain_cols):
            with col:
                d = col.selectbox(f"x{j+1}", options=[1, -1, 0],
                                  format_func=lambda v: {1: "≥ 0", -1: "≤ 0", 0: "livre"}[v],
                                  key=f"dom_{j}")
                domain_vals.append(d)

        st.subheader("Função Objetivo (Maximização)")
        obj_cols = st.columns(nv)
        obj_vals = []
        for j, col in enumerate(obj_cols):
            with col:
                v = col.number_input(f"c{j+1}", value=1.0, step=0.5, key=f"obj_{j}", format="%.2f")
                obj_vals.append(v)

        st.subheader("Restrições")
        cons_data = []
        for i in range(nc):
            row_cols = st.columns(nv + 2)
            coefs = []
            for j in range(nv):
                with row_cols[j]:
                    v = st.number_input(f"a{i+1}{j+1}", value=1.0, step=0.5,
                                        key=f"a_{i}_{j}", label_visibility="collapsed",
                                        format="%.2f")
                    coefs.append(v)
            with row_cols[nv]:
                sign = st.selectbox("sinal", ["<=", ">=", "=="], key=f"sign_{i}",
                                    label_visibility="collapsed")
            with row_cols[nv + 1]:
                rhs = st.number_input("RHS", value=4.0, step=0.5, key=f"rhs_{i}",
                                      label_visibility="collapsed", format="%.2f")
            cons_data.append((coefs, sign, rhs))

        # Build TXT from widgets
        lines = [str(nv), str(nc)]
        lines.append(" ".join(str(d) for d in domain_vals))
        lines.append(" ".join(str(v) for v in obj_vals))
        for coefs, sign, rhs in cons_data:
            lines.append(" ".join(str(c) for c in coefs) + f" {sign} {rhs}")
        generated_text = "\n".join(lines)

        with st.expander("📄 Arquivo gerado (TXT)"):
            st.code(generated_text, language="text")
        st.session_state.lp_text = generated_text

    # ── Text input ────────────────────────────────────────────────────────────
    else:
        examples = _list_examples()
        example_names = ["(nenhum)"] + [p.name for p in examples]
        selected = st.selectbox("📂 Carregar exemplo", example_names, key="example_select")
        if selected != "(nenhum)":
            st.session_state.lp_text = (EXAMPLES_DIR / selected).read_text(encoding="utf-8")

        st.session_state.lp_text = st.text_area(
            "Cole ou escreva o arquivo de entrada",
            value=st.session_state.lp_text or _example_text(),
            height=220,
            help="Formato: linha1=n_vars, linha2=n_cons, linha3=domínios, linha4=objetivo, linhas5+=restrições",
        )

    # ── Solve button ──────────────────────────────────────────────────────────
    st.divider()
    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        run_clicked = st.button("▶ Resolver", type="primary", use_container_width=True)
    with col_info:
        if st.session_state.result:
            r: SolveResult = st.session_state.result
            status_color = {"otimo": "🟢", "inviavel": "🔴", "ilimitado": "🟡"}.get(r.status, "⚪")
            st.markdown(f"**Status**: {status_color} `{r.status.upper()}`   |   "
                        f"**Z** = `{r.z:.{decimals}f}` " if r.z is not None else f"**Status**: {status_color} `{r.status.upper()}`")

    if run_clicked and st.session_state.lp_text.strip():
        with st.spinner("Resolvendo…"):
            diary: list[str] = []
            try:
                lp = parse_text(st.session_state.lp_text)
                fpi = to_fpi(lp)
                result = solve(fpi, policy=policy, decimals=decimals, digits=digits,
                               log_callback=lambda m: diary.append(m))
                st.session_state.result = result
                st.session_state.step_idx = 0
                st.session_state.diary = diary
                if st.session_state.llm_rationale:
                    st.session_state.diary.insert(0, "🤖 **Racional do Llama 3:**\n" + st.session_state.llm_rationale)
            except Exception as exc:
                st.error(f"Erro ao processar o problema: {exc}")
                st.session_state.result = None

    # ── Step-by-step tableau ──────────────────────────────────────────────────
    result: Optional[SolveResult] = st.session_state.result
    if result and result.tableau_history:
        st.divider()
        st.subheader("📊 Tableau – Modo História")
        max_step = len(result.tableau_history) - 1
        col_prev, col_slider, col_next = st.columns([1, 6, 1])
        with col_prev:
            if st.button("◀ Anterior") and st.session_state.step_idx > 0:
                st.session_state.step_idx -= 1
        with col_slider:
            st.session_state.step_idx = st.slider(
                "Passo", 0, max_step, st.session_state.step_idx, key="step_slider"
            )
        with col_next:
            if st.button("Próximo ▶") and st.session_state.step_idx < max_step:
                st.session_state.step_idx += 1

        idx = st.session_state.step_idx
        t = result.tableau_history[idx]
        b = result.basis_history[idx]
        t_str = format_tableau(t, result.var_names, b, decimals, digits)
        st.code(t_str, language="text")
        st.caption(f"Passo {idx}/{max_step}  |  Base: {[result.var_names[i] if i < len(result.var_names) else f'col{i}' for i in b]}")

    # ── Plotly 2D graph ───────────────────────────────────────────────────────
    if result and result.status == "otimo":
        try:
            lp_check = parse_text(st.session_state.lp_text)
            if lp_check.n_vars == 2:
                st.divider()
                st.subheader("📈 Região Factível (2 variáveis)")
                fig = _build_feasible_region_plot(lp_check, result, decimals)
                st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 – Llama 3 NLP
# ══════════════════════════════════════════════════════════════════════════════

with tab_llm:
    st.header("🤖 Parser de Linguagem Natural – Llama 3")
    st.markdown(
        "Cole a **descrição em linguagem natural** de um problema de PL. "
        "O Llama 3 (rodando localmente via Ollama) vai gerar automaticamente "
        "o arquivo de entrada e preencher a aba de Setup."
    )

    available_models = list_available_models()
    if available_models:
        st.success(f"Ollama detectado. Modelos disponíveis: {', '.join(available_models)}")
    else:
        st.warning("Ollama não detectado em `localhost:11434`. Certifique-se de que o servidor está rodando (`ollama serve`).")

    problem_description = st.text_area(
        "Enunciado do problema",
        height=200,
        placeholder=(
            "Exemplo: Uma fábrica produz dois produtos, A e B. Cada unidade de A "
            "gera lucro de R$50 e requer 2h de máquina e 1kg de material. "
            "Cada unidade de B gera R$80, 1h de máquina e 3kg de material. "
            "Disponível: 100h de máquina e 150kg de material. Maximize o lucro."
        ),
    )

    if st.button("🚀 Parsear com Llama 3", type="primary"):
        if not problem_description.strip():
            st.warning("Digite a descrição do problema primeiro.")
        else:
            with st.spinner(f"Consultando {llm_model} em localhost:11434 …"):
                llm_result = parse_natural_language(problem_description, model=llm_model)

            if llm_result.error:
                st.error(f"**Erro:** {llm_result.error}")
                if llm_result.raw_response:
                    with st.expander("Resposta bruta do modelo"):
                        st.text(llm_result.raw_response)
            else:
                st.success("✅ Arquivo gerado com sucesso!")
                col_rat, col_file = st.columns(2)
                with col_rat:
                    st.subheader("Racional de Modelagem")
                    st.info(llm_result.rationale)
                with col_file:
                    st.subheader("Arquivo TXT Gerado")
                    st.code(llm_result.lp_file, language="text")

                st.session_state.lp_text = llm_result.lp_file
                st.session_state.llm_rationale = llm_result.rationale
                st.toast("Arquivo copiado para a aba de Setup! Clique em 'Resolver'.", icon="✅")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 – Diary / Logs
# ══════════════════════════════════════════════════════════════════════════════

with tab_diary:
    st.header("📓 Diário Explicativo da Resolução")

    result: Optional[SolveResult] = st.session_state.result
    diary: list[str] = st.session_state.get("diary", [])

    if not diary:
        st.info("Nenhuma resolução executada ainda. Vá à aba **Setup** e clique em **Resolver**.")
    else:
        if result:
            badge = {"otimo": "🟢 ÓTIMO", "inviavel": "🔴 INVIÁVEL", "ilimitado": "🟡 ILIMITADO"}.get(result.status, result.status)
            st.markdown(f"### Resultado Final: {badge}")
            if result.status == "otimo":
                st.markdown(f"**Z ótimo** = `{result.z:.{decimals}f}`")

                cols = st.columns(2)
                with cols[0]:
                    st.markdown("**Solução Primal**")
                    for j, name in enumerate(result.var_names):
                        st.markdown(f"- `{name}` = **{result.x_primal[j]:.{decimals}f}**")
                with cols[1]:
                    st.markdown("**Solução Dual (preços-sombra)**")
                    for i, yi in enumerate(result.y_dual):
                        st.markdown(f"- `y{i+1}` = **{yi:.{decimals}f}**")

                st.divider()
                _render_economic_interpretation(result, decimals)

        st.divider()
        st.subheader("📜 Log Detalhado")

        filter_text = st.text_input("🔍 Filtrar logs", placeholder="Ex: Tableau, entra, Fase")
        show_tableaux = st.checkbox("Mostrar tableaux nos logs", value=False)

        filtered = []
        for entry in diary:
            if filter_text and filter_text.lower() not in entry.lower():
                continue
            if not show_tableaux and "Tableau" in entry and "\n" in entry:
                lines_e = entry.split("\n")
                filtered.append(lines_e[0] + "  *(tableau omitido)*")
                continue
            filtered.append(entry)

        log_text = "\n\n".join(filtered)
        st.text_area("", value=log_text, height=500, key="log_display", disabled=True)

        if st.button("💾 Exportar log (.txt)"):
            st.download_button(
                label="⬇ Download log",
                data="\n\n".join(diary),
                file_name="suplex_log.txt",
                mime="text/plain",
            )


