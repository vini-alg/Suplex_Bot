#!/usr/bin/env bash

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$REPO_DIR/.venv"

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

# ── Helpers ───────────────────────────────────────────────────────────────────
ok()   { echo -e "${GREEN}  ✔  $*${NC}"; }
info() { echo -e "${CYAN}  →  $*${NC}"; }
warn() { echo -e "${YELLOW}  ⚠  $*${NC}"; }
err()  { echo -e "${RED}  ✗  $*${NC}"; }

header() {
    echo ""
    echo -e "${BOLD}${BLUE}╔══════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}${BLUE}║          Suplex Bot – Setup Interativo               ║${NC}"
    echo -e "${BOLD}${BLUE}╚══════════════════════════════════════════════════════╝${NC}"
    echo ""
}

# ── Task: Python venv + deps ──────────────────────────────────────────────────
install_python() {
    echo ""
    info "Criando ambiente virtual em $VENV_DIR"
    python3 -m venv "$VENV_DIR"
    info "Instalando dependências Python..."
    "$VENV_DIR/bin/pip" install --upgrade pip --quiet
    "$VENV_DIR/bin/pip" install -r "$REPO_DIR/requirements.txt"
    ok "Dependências Python instaladas."
}

# ── Task: Ensure Ollama binary ────────────────────────────────────────────────
ensure_ollama() {
    if command -v ollama &>/dev/null; then
        ok "Ollama já instalado."
        return 0
    fi
    info "Ollama não encontrado."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        info "Instalando Ollama (Linux)..."
        curl -fsSL https://ollama.com/install.sh | sh
        ok "Ollama instalado."
    else
        warn "Instale o Ollama manualmente: https://ollama.com/download"
        warn "Depois execute: bash setup.sh → opção 3"
        return 1
    fi
}

# ── Task: Model selection + pull ─────────────────────────────────────────────
MODEL_MENU=(
    "llama3      – Recomendado        (4.7 GB, ~8 GB RAM)"
    "mistral     – Mais rápido        (4.1 GB, ~6 GB RAM)"
    "phi3        – Máquinas modestas  (1.6 GB, ~4 GB RAM)"
    "gemma2      – Google             (5.5 GB, ~8 GB RAM)"
    "codellama   – Focado em código   (3.8 GB, ~6 GB RAM)"
    "Outro...    – Digitar nome do modelo"
)
MODEL_IDS=("llama3" "mistral" "phi3" "gemma2" "codellama" "__custom__")

pick_and_pull_model() {
    ensure_ollama || return 1

    echo ""
    echo -e "${BOLD}  Escolha o modelo para instalar:${NC}"
    for i in "${!MODEL_MENU[@]}"; do
        printf "  ${CYAN}%d)${NC} %s\n" $((i+1)) "${MODEL_MENU[$i]}"
    done
    echo ""

    local choice model
    while true; do
        read -rp "  Opção [1-${#MODEL_MENU[@]}]: " choice
        if [[ "$choice" =~ ^[0-9]+$ ]] && (( choice >= 1 && choice <= ${#MODEL_MENU[@]} )); then
            break
        fi
        warn "Opção inválida, tente novamente."
    done

    local idx=$(( choice - 1 ))
    model="${MODEL_IDS[$idx]}"

    if [[ "$model" == "__custom__" ]]; then
        read -rp "  Nome do modelo (ex: llama3.1, deepseek-r1): " model
        [[ -z "$model" ]] && err "Nome vazio, cancelado." && return 1
    fi

    # Ensure server is reachable before pulling
    if ! curl -sf http://localhost:11434 &>/dev/null; then
        info "Iniciando servidor Ollama em background..."
        ollama serve &>/dev/null &
        sleep 2
    fi

    info "Baixando '$model' (pode demorar – tamanho indicado acima)..."
    if ollama pull "$model"; then
        ok "Modelo '$model' instalado."
    else
        err "Falha ao baixar '$model'. Verifique o nome e tente novamente."
        return 1
    fi
}

# ── Task: Uninstall ───────────────────────────────────────────────────────────
uninstall_all() {
    echo ""
    warn "Isso irá:"
    warn "  • Remover o ambiente virtual  (.venv)"
    warn "  • Remover TODOS os modelos Ollama instalados localmente"
    echo ""
    read -rp "  Confirma? [s/N]: " confirm
    [[ "${confirm,,}" != "s" ]] && info "Cancelado." && return

    # Remove .venv
    if [[ -d "$VENV_DIR" ]]; then
        rm -rf "$VENV_DIR"
        ok ".venv removido."
    else
        info ".venv não encontrado."
    fi

    # Remove ollama models
    if command -v ollama &>/dev/null; then
        mapfile -t installed < <(ollama list 2>/dev/null | awk 'NR>1 && NF>0 {print $1}')
        if [[ ${#installed[@]} -eq 0 ]]; then
            info "Nenhum modelo Ollama instalado."
        else
            for m in "${installed[@]}"; do
                info "Removendo modelo: $m"
                ollama rm "$m" && ok "  $m removido." || warn "  Falha ao remover $m."
            done
        fi
    else
        info "Ollama não instalado – nenhum modelo para remover."
    fi
    ok "Limpeza concluída."
}

# ── Summary banner ────────────────────────────────────────────────────────────
print_summary() {
    echo ""
    echo -e "${GREEN}${BOLD}╔══════════════════════════════════════════════════════╗"
    echo -e "║  ✅  Suplex Bot – pronto!                            ║"
    echo -e "╠══════════════════════════════════════════════════════╣"
    echo -e "║  Ativar venv  : source .venv/bin/activate            ║"
    echo -e "║  Iniciar app  : bash run_server.sh                   ║"
    echo -e "║  CLI          : python simplex_solver.py <arquivo>   ║"
    echo -e "╠══════════════════════════════════════════════════════╣"
    echo -e "║  LLM – outros modelos:                               ║"
    echo -e "║    ollama pull mistral   (4.1 GB – mais rápido)      ║"
    echo -e "║    ollama pull phi3      (1.6 GB – máquinas modestas)║"
    echo -e "╚══════════════════════════════════════════════════════╝${NC}"
    echo ""
}

# ── Main menu ─────────────────────────────────────────────────────────────────
header

echo -e "  ${BOLD}O que deseja fazer?${NC}"
echo ""
echo -e "  ${CYAN}1)${NC} Instalar tudo              (Python deps + Ollama + modelo LLM)"
echo -e "  ${CYAN}2)${NC} Apenas dependências Python (.venv + requirements.txt)"
echo -e "  ${CYAN}3)${NC} Apenas modelo LLM          (escolher e baixar via Ollama)"
echo -e "  ${CYAN}4)${NC} Remover tudo               (apagar .venv e modelos locais)"
echo -e "  ${CYAN}0)${NC} Sair"
echo ""

read -rp "  Opção [0-4]: " CHOICE

case "$CHOICE" in
    1)
        install_python
        pick_and_pull_model
        print_summary
        ;;
    2)
        install_python
        print_summary
        ;;
    3)
        pick_and_pull_model
        print_summary
        ;;
    4)
        uninstall_all
        ;;
    0)
        info "Saindo."
        ;;
    *)
        err "Opção inválida."
        exit 1
        ;;
esac
