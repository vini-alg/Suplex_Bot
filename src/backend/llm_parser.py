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

# ─────────────────────────── System prompt ───────────────────────────────────

_SYSTEM_PROMPT = """\
Você é um especialista em Pesquisa Operacional que converte problemas de \
Programação Linear (PL) descritos em linguagem natural para um formato \
numérico estruturado.

FORMATO DE SAÍDA OBRIGATÓRIO – responda APENAS com um objeto JSON válido \
contendo exatamente as duas chaves abaixo, sem qualquer texto fora do JSON:

{
  "rationale": "<parágrafo curto explicando suas escolhas de modelagem>",
  "lp_lines": ["<linha1>", "<linha2>", ...]
}

A chave "lp_lines" é um ARRAY JSON onde cada elemento é uma linha do arquivo TXT:
  Índice 0 : número de variáveis de decisão (inteiro)
  Índice 1 : número de restrições (inteiro)
  Índice 2 : domínio das variáveis, separado por espaços (1=x≥0, -1=x≤0, 0=livre)
  Índice 3 : coeficientes da função objetivo (MAXIMIZAÇÃO), separados por espaços
  Índices 4+: cada restrição no formato "<coef1> <coef2> ... <= <RHS>"
              use <= , >= ou == conforme o problema

EXEMPLO de saída para "Maximizar 5x1+4x2 sujeito a 6x1+x2<=6 e x1+x2<=4":
{
  "rationale": "x1 e x2 são os produtos; domínio não-negativo; dois recursos.",
  "lp_lines": ["2", "2", "1 1", "5 4", "6 1 <= 6", "1 1 <= 4"]
}

REGRAS:
- Se o problema pede minimização, inverta os sinais dos coeficientes da \
  função objetivo (transforme em maximização equivalente).
- Identifique claramente cada variável de decisão e documente no "rationale".
- Todos os números devem ser inteiros ou decimais (ponto como separador decimal).
- Use somente <=, >= ou == como operadores de restrição.
- NÃO inclua texto algum fora do bloco JSON. NÃO use newlines dentro de strings JSON.
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

    try:
        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": problem_text},
            ],
            options={"temperature": 0.1},
        )
        # ollama >=0.6 returns an object; older versions return a dict
        msg = response.message if hasattr(response, "message") else response["message"]
        raw = msg.content if hasattr(msg, "content") else msg["content"]
    except Exception as exc:
        return LLMResult(
            lp_file="", rationale="", raw_response="",
            error=f"Erro ao contatar Ollama ({OLLAMA_HOST}): {exc}",
        )

    # ── Parse JSON from response ──────────────────────────────────────────────
    try:
        cleaned = _extract_json(raw)
        data = json.loads(cleaned)

        # Prefer lp_lines (array) over lp_file (legacy string)
        if "lp_lines" in data:
            lp_file = "\n".join(str(l) for l in data["lp_lines"])
        else:
            lp_file = data.get("lp_file", "").strip()

        return LLMResult(
            lp_file=lp_file,
            rationale=data.get("rationale", "").strip(),
            raw_response=raw,
        )
    except (json.JSONDecodeError, KeyError) as exc:
        return LLMResult(
            lp_file="", rationale="",
            raw_response=raw,
            error=f"Falha ao interpretar JSON do modelo: {exc}\n\nResposta bruta:\n{raw}",
        )


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
