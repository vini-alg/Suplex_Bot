# Project Suplex — Simplex Optimization

<p align="center">
  <img src="doc/src/big_icon.png" width="320" alt="Project Suplex – Simplex Optimization"/>
</p>

Solucionador do **Método Simplex** (Fase I + Fase II) com interface web interativa e integração com LLM local (Llama 3 via Ollama). Interpreta enunciados em linguagem natural e resolve problemas de Programação Linear com visualização gráfica da região factível.

## Stack

| Camada | Tecnologia |
|--------|-----------|
| Core | Python 3, NumPy (pivoteamento manual — sem scipy) |
| CLI | `argparse` |
| UI | Streamlit + Plotly |
| LLM | Qualquer modelo via [Ollama](https://ollama.com) self-hosted |
| Testes | pytest (unitários + integração) |

---

## 1. Instalação

```bash
git clone https://github.com/vini-alg/Suplex_Bot.git
cd Suplex_Bot
chmod +x setup.sh
bash setup.sh          # cria .venv, instala dependências e configura Ollama
source .venv/bin/activate
```

O `setup.sh` oferece um menu interativo para instalar o Ollama, baixar modelos e iniciar o servidor.

---

## 2. Iniciar a aplicação

```bash
chmod +x run_server.sh # Se é a primeira vez que está executando o script
bash run_server.sh
```

O script detecta automaticamente o Ollama e o modelo instalado, e sobe o servidor Streamlit em `http://localhost:8501`.

Para iniciar manualmente:

```bash
# 1. Servidor Ollama (se não estiver rodando)
ollama serve

# 2. Interface Streamlit
streamlit run src/interface/app.py
```

---

## 3. Configuração do LLM (Ollama)

### 3.1 Instalar o Ollama

- **Linux**: `curl -fsSL https://ollama.com/install.sh | sh`
- **Mac / Windows**: [ollama.com/download](https://ollama.com/download)

### 3.2 Modelos suportados

```bash
ollama pull llama3      # recomendado (4.7 GB, ~8 GB RAM) e atualmente testado
ollama pull mistral     # mais rápido (4.1 GB) não testado
ollama pull phi3        # máquinas modestas (1.6 GB) não testado
```

> O modelo padrão é `llama3`. Altere pelo campo **"Modelo Ollama"** na barra lateral da UI ou via variável de ambiente `OLLAMA_MODEL`.

### 3.3 Aceleração GPU

O Ollama usa GPU automaticamente se os drivers estiverem instalados (CUDA para NVIDIA, ROCm para AMD). Verifique com `ollama ps`.

---

## 4. Interface Web

Acesse `http://localhost:8501` após iniciar o servidor.

| Aba | Funcionalidades |
|-----|----------------|
| 🔧 **Setup & Visualização** | Entrada manual via sliders **ou** texto estruturado com carregamento de exemplos; passo a passo do tableau; gráfico Plotly da região factível com vértices e caminho do simplex (problemas 2D) |
| 🤖 **LLM (NLP)** | Cole um enunciado em linguagem natural; pipeline de 2 estágios (interpretação → JSON) com fallback automático de reparo; saída carregada diretamente no solver |
| 📓 **Diário / Logs** | Log completo da resolução + interpretação econômica dos preços-sombra |

### Pipeline LLM (2 estágios + reparo)

```
Estágio 1 – Modelagem
  Texto natural → modelo matemático normalizado (MAX, coeficientes numéricos, zeros explícitos)

Estágio 2 – Formatação JSON
  Modelo matemático → {"lp_lines": [...]}

Reparo (fallback automático)
  Se o JSON do estágio 2 for inválido, um terceiro prompt com ambas as saídas
  anteriores é enviado para auto-correção antes de retornar erro.
```

Parâmetros LLM fixos para saída determinística: `temperature=0`, `seed=42`, `top_k=1`, `top_p=1.0`.

---

## 5. CLI

```bash
python simplex_solver.py <arquivo.txt> [opções]

# Exemplos
python simplex_solver.py examples/example_2var.txt
python simplex_solver.py examples/example_multi_vertex.txt --policy bland --decimals 6
python simplex_solver.py examples/example_phase1.txt --digits 12
```

| Argumento | Descrição | Padrão |
|-----------|-----------|--------|
| `filename` | Arquivo de entrada `.txt` | — |
| `--policy` | Regra de pivô: `largest`, `bland`, `smallest` | `largest` |
| `--decimals` | Casas decimais na saída | `4` |
| `--digits` | Largura de coluna no tableau | `10` |

---

## 6. Formato do arquivo de entrada

```
<n_vars>
<n_cons>
<domínio: 1=x≥0  -1=x≤0  0=livre>   ← um valor por variável
<coeficientes da FO – sempre MAXIMIZAÇÃO>
<coef1 ... coefN> <= / >= / == <RHS>  ← uma linha por restrição
...
```

Exemplo (`examples/example_2var.txt`) — Maximizar 5x₁ + 4x₂:

```
2
2
1 1
5 4
6 1 <= 6
1 1 <= 4
```

---

## 7. Exemplos incluídos

| Arquivo | Descrição | Resultado esperado |
|---------|-----------|-------------------|
| `example_2var.txt` | 2 vars, 2 restrições ≤ | Ótimo Z=5.2 |
| `example_3var.txt` | 3 vars, restrições mistas | Ótimo |
| `example_phase1.txt` | Restrições ≥ / == (exige Fase I) | Ótimo via Fase I+II |
| `example_multi_vertex.txt` | 2 vars, 4 restrições — 5 vértices | Ideal para visualizar o gráfico |

---

## 8. Testes

```bash
# Solver (unitários — sem dependências externas)
python -m pytest tests/test_solver.py -v

# Parser LLM (mocked — sem Ollama)
python -m pytest tests/test_llm_parser.py -v -m "not integration"

# Integração LLM (requer Ollama rodando)
python -m pytest tests/test_llm_parser.py -v -m integration
```

Cobertura dos testes:

| Módulo | Testes |
|--------|--------|
| `test_solver.py` | Ótimo (pequeno / médio / grande), Inviável, Ilimitado, 3 regras de pivô |
| `test_llm_parser.py` | Pipeline mocked (8 casos), reparo de JSON, integração com Ollama (3 casos) |

---

## 9. Sujestões

- Teste novos exemplos para verificar a robustez do sistema.
- Tente usar outros modelos do Ollama para melhorar a precisão da interpretação.
- Ao rodar, tente ajustar o prompt do LLM para melhorar a interpretação do enunciado. Localizado em `src/backend/llm_parser.py` linha 29.

## 10. Estrutura do Projeto

```
Suplex_Bot/
├── run_server.sh               ← lança Ollama + Streamlit automaticamente
├── setup.sh                    ← setup interativo (.venv, Ollama, modelos)
├── requirements.txt
├── pytest.ini
├── simplex_solver.py           ← CLI entry point
├── examples/
│   ├── example_2var.txt
│   ├── example_3var.txt
│   ├── example_phase1.txt
│   └── example_multi_vertex.txt
├── tests/
│   ├── test_solver.py          ← testes unitários do Simplex
│   └── test_llm_parser.py      ← testes do pipeline LLM
├── doc/src/
│   ├── big_icon.png            ← logo completo
│   └── small_icon.png          ← ícone circular (UI)
└── src/
    ├── backend/
    │   ├── utils.py            ← parser de arquivo + conversão FPI
    │   ├── suplex.py           ← Simplex Fase I + Fase II (pivoteamento manual)
    │   └── llm_parser.py       ← pipeline LLM 2 estágios + reparo
    └── interface/
        └── app.py              ← UI Streamlit
```