#!/usr/bin/env bash
#
# run_server.sh – Start Suplex Bot
#   1. Ensures the Ollama daemon is running
#   2. Detects installed models; asks which to use when >1
#   3. Exports OLLAMA_MODEL so llm_parser.py picks it up
#   4. Starts Streamlit

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$REPO_DIR/.venv"

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

ok()   { echo -e "${GREEN}  ✔  $*${NC}"; }
info() { echo -e "${CYAN}  →  $*${NC}"; }
warn() { echo -e "${YELLOW}  ⚠  $*${NC}"; }
err()  { echo -e "${RED}  ✗  $*${NC}"; }

echo ""
echo -e "${BOLD}${BLUE}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${BLUE}║              Suplex Bot – Launcher                  ║${NC}"
echo -e "${BOLD}${BLUE}╚══════════════════════════════════════════════════════╝${NC}"
echo ""

# ── 1. Check Python venv ──────────────────────────────────────────────────────
STREAMLIT="$VENV_DIR/bin/streamlit"
if [[ ! -x "$STREAMLIT" ]]; then
    err ".venv não encontrado ou incompleto."
    warn "Execute primeiro: bash setup.sh → opção 1 ou 2"
    exit 1
fi

# ── 2. Check Ollama installed ─────────────────────────────────────────────────
if ! command -v ollama &>/dev/null; then
    err "Ollama não encontrado."
    warn "Execute: bash setup.sh → opção 1 ou 3"
    exit 1
fi

# ── 3. Ensure Ollama daemon is running ────────────────────────────────────────
if curl -sf http://localhost:11434 &>/dev/null; then
    ok "Servidor Ollama já está rodando."
else
    info "Iniciando servidor Ollama em background..."
    ollama serve &>/dev/null &
    OLLAMA_PID=$!
    # Wait until responsive (up to 10 s)
    for i in $(seq 1 10); do
        sleep 1
        if curl -sf http://localhost:11434 &>/dev/null; then
            ok "Servidor Ollama iniciado (PID $OLLAMA_PID)."
            break
        fi
        if [[ $i -eq 10 ]]; then
            err "Servidor Ollama não respondeu em 10 s."
            kill "$OLLAMA_PID" 2>/dev/null
            exit 1
        fi
    done
fi

# ── 4. Detect installed models ────────────────────────────────────────────────
mapfile -t MODELS < <(ollama list 2>/dev/null | awk 'NR>1 && NF>0 {print $1}')

if [[ ${#MODELS[@]} -eq 0 ]]; then
    err "Nenhum modelo Ollama instalado."
    warn "Execute: bash setup.sh → opção 3"
    warn "A aba NLP não estará disponível."
    echo ""
    read -rp "  Continuar sem LLM? [s/N]: " cont
    [[ "${cont,,}" != "s" ]] && exit 1
    export OLLAMA_MODEL=""
else
    # ── 5. Check which model is currently loaded in memory ────────────────────
    mapfile -t RUNNING < <(ollama ps 2>/dev/null | awk 'NR>1 && NF>0 {print $1}')

    if [[ ${#RUNNING[@]} -gt 0 ]]; then
        CHOSEN="${RUNNING[0]}"
        ok "Modelo já em memória: ${BOLD}${CHOSEN}${NC}"

        if [[ ${#MODELS[@]} -gt 1 ]]; then
            echo ""
            read -rp "  Usar outro modelo? [s/N]: " swap
            if [[ "${swap,,}" == "s" ]]; then
                CHOSEN=""
            fi
        fi
    fi

    # ── 6. Ask user if more than 1 installed and none chosen yet ─────────────
    if [[ -z "$CHOSEN" ]]; then
        if [[ ${#MODELS[@]} -eq 1 ]]; then
            CHOSEN="${MODELS[0]}"
            ok "Único modelo instalado: ${BOLD}${CHOSEN}${NC}"
        else
            echo ""
            echo -e "  ${BOLD}Modelos instalados:${NC}"
            for i in "${!MODELS[@]}"; do
                printf "  ${CYAN}%d)${NC} %s\n" $((i+1)) "${MODELS[$i]}"
            done
            echo ""

            while true; do
                read -rp "  Escolha o modelo para esta sessão [1-${#MODELS[@]}]: " choice
                if [[ "$choice" =~ ^[0-9]+$ ]] && (( choice >= 1 && choice <= ${#MODELS[@]} )); then
                    break
                fi
                warn "Opção inválida, tente novamente."
            done

            CHOSEN="${MODELS[$(( choice - 1 ))]}"
            ok "Modelo selecionado: ${BOLD}${CHOSEN}${NC}"
        fi
    fi

    export OLLAMA_MODEL="$CHOSEN"
fi

# ── 7. Start Streamlit ────────────────────────────────────────────────────────
echo ""
echo -e "  ${BOLD}Modelo LLM :${NC} ${OLLAMA_MODEL:-"(nenhum)"}"
echo -e "  ${BOLD}URL        :${NC} http://localhost:8501"
echo ""
info "Iniciando Streamlit... (Ctrl+C para parar)"
echo ""

exec "$STREAMLIT" run "$REPO_DIR/src/interface/app.py"
