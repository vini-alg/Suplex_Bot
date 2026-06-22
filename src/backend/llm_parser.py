"""
llm_parser.py – Local Llama 3 integration via Ollama API.

Sends a natural-language LP problem description to the local Ollama server
and returns both:
  - The structured TXT file content (ready for parse_file / parse_text)
  - A short modelling rationale paragraph (for the Diary tab)
"""

from __future__ import annotations

import re
import json
from dataclasses import dataclass
from typing import Optional

try:
    import ollama
    _OLLAMA_AVAILABLE = True
except ImportError:
    _OLLAMA_AVAILABLE = False

import os as _os
OLLAMA_MODEL = _os.environ.get("OLLAMA_MODEL", "llama3")
OLLAMA_HOST  = _os.environ.get("OLLAMA_HOST",  "http://localhost:11434")

# ─────────────────────────── Stage 1 prompt: understand & simplify ───────────

_PROMPT_SIMPLIFY = """\
Você é um modelador matemático determinístico.
Sua única tarefa é ler um problema de Programação Linear e convertê-lo ESTRITAMENTE para um modelo matemático puro.
NÃO escreva o significado das variáveis. NÃO adicione texto explicativo entre parênteses. 

REGRA DE ALINHAMENTO (CRÍTICA): A função objetivo e todas as restrições devem conter TODAS as variáveis do problema. Se uma variável não fizer parte de uma linha, declare-a explicitamente multiplicada por zero (ex: + 0*x3). 
DICA DE IGUALDADE: Se o texto disser que A = B, converta para 1*A - 1*B = 0.

RESPONDA APENAS com o bloco abaixo. Substitua os valores, mas mantenha as tags [OBJ], [ST] e [BOUNDS] intactas:

-NUM_VAR:
x1, x2, ..., xN

-OBJ:
MAX or MIN: <coef>*x1 + <coef>*x2 + ...

-BOUNDS:
[x1 >= 0]
[x2 <= 0]
[x3 livre]

-ST:
[<coef>*x1 + <coef>*x2 + ... <= <rhs>]
[<coef>*x1 + <coef>*x2 + ... == <rhs>]
[<coef>*x1 + <coef>*x2 + ... >= <rhs>]
REGRA CRÍTICA: Lembre-se sempre de deixar todas as variaveis à esquerda da igualdade/inequalidade.
"""

# ─────────────────────────── Stage 2 prompt: format only ─────────────────────

_PROMPT_FORMAT = """\
Você é um parser de dados. Sua única tarefa é converter a descrição matemática de um problema de PL ESTRITAMENTE para o formato JSON abaixo.
NÃO pense, NÃO explique, NÃO use blocos de código Markdown (```). APENAS produza o objeto JSON começando com { e terminando com }.

FORMATO OBRIGATÓRIO (Um array de strings):
{"lp_lines": ["<linha0>", "<linha1>", "<linha2>", "<linha3>", "<linha4>"]}

Regras de cada índice do array "lp_lines":
  Índice 0: APENAS o número total de variáveis (ex: "3").
  Índice 1: APENAS o número total de restrições (ex: "3").
  Índice 2: Domínio das variáveis separados por espaço (1 = x>=0; -1 = x<=0; 0 = livre). Ex: "1 -1 0".
  Índice 3: Coeficientes da função objetivo separados por espaço. Ex: "40 30 60".
  Índices 4+: Cada restrição no formato exato "<coef1> <coef2> ... <sinal> <rhs>". Ex: "1 1 0 == 10000".

EXEMPLO DE SAÍDA ESPERADA:
{"lp_lines": ["3", "3", "1 0 -1", "40 30 60", "20 -3 1 <= 15000", "1 -1 0 == 10000", "0 0 1 >= 2000"]}

REGRAS CRÍTICAS:
- Extraia apenas os NÚMEROS dos coeficientes das restrições. Nunca escreva "x1" ou "*". 
- Mantenha os coeficientes à esquerda da inequalidade/igualdade.
- Preencha para cada variavel não presente na linha com 0.
"""

# ─────────────────────────── Result dataclass ────────────────────────────────

@dataclass
class LLMResult:
    lp_file: str
    rationale: str
    raw_response: str
    error: Optional[str] = None


# ─────────────────────────── Main function ───────────────────────────────────

def parse_natural_language(problem_text: str, model: str = OLLAMA_MODEL) -> LLMResult:
    """
    Send `problem_text` to the local Llama 3 model and return a LLMResult.

    Parameters
    ----------
    problem_text : str
        Natural language description of the LP problem.
    model : str
        Ollama model tag (default: 'llama3').

    Returns
    -------
    LLMResult with fields: lp_file, rationale, raw_response, error
    """
    if not _OLLAMA_AVAILABLE:
        return LLMResult(
            lp_file="", rationale="",
            raw_response="",
            error="Biblioteca 'ollama' não instalada. Execute: pip install ollama",
        )

    client = ollama.Client(host=OLLAMA_HOST)

    # ── Stage 1: understand & simplify (plain text, no JSON pressure) ─────────
    rationale, err1 = _call_ollama(
        client, model, _PROMPT_SIMPLIFY, problem_text, temperature=0.3
    )
    if err1:
        return LLMResult(lp_file="", rationale="", raw_response="", error=err1)

    # ── Stage 2: format only (deterministic, strictly JSON) ───────────────────
    raw, err2 = _call_ollama(
        client, model, _PROMPT_FORMAT, rationale, temperature=0.0
    )
    if err2:
        return LLMResult(lp_file="", rationale=rationale, raw_response="", error=err2)

    # ── Parse JSON from stage-2 response ──────────────────────────────────────
    try:
        data = json.loads(_extract_json(raw))
        if "lp_lines" in data:
            lp_file = "\n".join(str(l) for l in data["lp_lines"])
        else:
            lp_file = data.get("lp_file", "").strip()
        return LLMResult(lp_file=lp_file, rationale=rationale, raw_response=raw)
    except (json.JSONDecodeError, KeyError) as exc:
        return LLMResult(
            lp_file="", rationale=rationale, raw_response=raw,
            error=f"Falha ao interpretar JSON (etapa 2): {exc}\n\nResposta bruta:\n{raw}",
        )


def _call_ollama(
    client: "ollama.Client",
    model: str,
    system_prompt: str,
    user_text: str,
    temperature: float = 0.1,
) -> tuple[str, Optional[str]]:
    """Call Ollama chat and return (content, error). error is None on success."""
    try:
        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_text},
            ],
            options={"temperature": temperature},
        )
        msg = response.message if hasattr(response, "message") else response["message"]
        content = msg.content if hasattr(msg, "content") else msg["content"]
        return content, None
    except Exception as exc:
        return "", f"Erro ao contatar Ollama ({OLLAMA_HOST}): {exc}"


def _extract_json(text: str) -> str:
    """Strip markdown code fences, sanitize control chars, return JSON content."""
    text = text.strip()
    # Remove ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        text = match.group(1).strip()
    else:
        # Find first { ... } block
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start != -1 and end > start:
            text = text[start:end]

    # Sanitize literal control characters (newlines/tabs) inside JSON strings
    # so json.loads doesn't fail with "Invalid control character"
    text = re.sub(
        r'"((?:[^"\\]|\\.)*?)"',
        lambda m: '"' + m.group(1).replace('\n', '\\n').replace('\r', '').replace('\t', '\\t') + '"',
        text,
    )
    return text


def list_available_models() -> list[str]:
    """Return list of models available in the local Ollama instance."""
    if not _OLLAMA_AVAILABLE:
        return []
    try:
        client = ollama.Client(host=OLLAMA_HOST)
        response = client.list()
        # ollama >=0.6 returns Pydantic objects; older versions return dicts
        items = response.models if hasattr(response, "models") else response.get("models", [])
        return [
            (m.model if hasattr(m, "model") else m.get("name", str(m)))
            for m in items
        ]
    except Exception:
        return []
