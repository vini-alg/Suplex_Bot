"""
tests/test_llm_parser.py – Tests for the LLM parsing pipeline.

Strategy
--------
Rather than calling the real Ollama server (slow, non-deterministic, requires
the daemon to be running), we use unittest.mock to inject a controlled
stage-1 / stage-2 / stage-3 response so every test is fast and reproducible.

Two test classes:

  TestParseNaturalLanguageMocked
      Patches _call_ollama to return pre-written intermediate outputs and
      verifies that the final lp_file string matches the expected format
      accepted by parse_text / parse_file.

  TestLLMIntegration (marked with @pytest.mark.integration)
      Actually calls the Ollama server.  Skipped automatically when Ollama is
      not reachable.  Run explicitly with:
          pytest tests/test_llm_parser.py -v -m integration

Run unit tests only:
    python -m pytest tests/test_llm_parser.py -v
"""

import sys
import os
import json
import pytest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.backend.llm_parser import parse_natural_language, LLMResult
from src.backend.utils import parse_text


# ─────────────────────────── mock helpers ────────────────────────────────────

def _make_mock_calls(stage1_text: str, stage2_json: dict):
    """
    Returns a side_effect list for _call_ollama covering the 2-stage happy path.
    Each entry is (content, error) matching the helper's return signature.
    """
    return [
        (stage1_text, None),
        (json.dumps(stage2_json), None),
    ]


# ─────────────────────────── MOCKED UNIT TESTS ───────────────────────────────

class TestParseNaturalLanguageMocked:

    # ── 2-variable optimal ────────────────────────────────────────────────────
    def test_small_2var_lp(self):
        """
        'Maximise 5x1 + 4x2 subject to 6x1+x2<=6 and x1+x2<=4, x1,x2>=0'
        Expected lp_file:
            2
            2
            1 1
            5 4
            6 1 <= 6
            1 1 <= 4
        """
        stage3 = {"lp_lines": ["2", "2", "1 1", "5 4", "6 1 <= 6", "1 1 <= 4"]}
        calls = _make_mock_calls("(stage1 placeholder)", stage3)

        with patch("src.backend.llm_parser._call_ollama", side_effect=calls):
            result = parse_natural_language("Maximise 5x1+4x2 s.t. 6x1+x2<=6 and x1+x2<=4")

        assert result.error is None
        lp = parse_text(result.lp_file)
        assert lp.n_vars == 2
        assert lp.n_cons == 2
        assert list(lp.domain) == [1, 1]
        assert lp.c.tolist() == pytest.approx([5.0, 4.0])
        assert lp.signs == ["<=", "<="]

    # ── 3-variable with equality and lower bound ──────────────────────────────
    def test_medium_3var_with_equality(self):
        """
        Maximise 40x1+30x2+60x3
        x1+x2+x3<=10000; x3>=2000; x1==x2; 2x1+3x2+x3<=15000
        x1,x2,x3 >= 0
        """
        stage3 = {
            "lp_lines": [
                "3", "4", "1 1 1", "40 30 60",
                "1 1 1 <= 10000",
                "0 0 1 >= 2000",
                "-1 1 0 == 0",
                "2 3 1 <= 15000",
            ]
        }
        calls = _make_mock_calls("(stage1)", stage3)

        with patch("src.backend.llm_parser._call_ollama", side_effect=calls):
            result = parse_natural_language("Medium 3-var problem")

        assert result.error is None
        lp = parse_text(result.lp_file)
        assert lp.n_vars == 3
        assert lp.n_cons == 4
        assert lp.signs == ["<=", ">=", "==", "<="]
        assert lp.c.tolist() == pytest.approx([40.0, 30.0, 60.0])

    # ── domain flags: mixed (>=0, <=0, free) ─────────────────────────────────
    def test_mixed_domain(self):
        """
        Tests that domain -1 (x<=0) and 0 (free) are preserved through parsing.
        """
        stage3 = {
            "lp_lines": [
                "3", "2", "1 -1 0", "2 3 5",
                "3 2 5 <= 15",
                "2 3 0 <= 10",
            ]
        }
        calls = _make_mock_calls("(s1)", stage3)

        with patch("src.backend.llm_parser._call_ollama", side_effect=calls):
            result = parse_natural_language("Mixed domain problem")

        assert result.error is None
        lp = parse_text(result.lp_file)
        assert list(lp.domain) == [1, -1, 0]

    # ── rationale is stage-1 output ───────────────────────────────────────────
    def test_rationale_comes_from_stage1(self):
        stage1_text = "Stage-1 math description here."
        stage3 = {"lp_lines": ["2", "1", "1 1", "1 1", "1 1 <= 5"]}
        calls = _make_mock_calls(stage1_text, stage3)

        with patch("src.backend.llm_parser._call_ollama", side_effect=calls):
            result = parse_natural_language("Simple problem")

        assert result.rationale == stage1_text

    # ── stage-1 failure propagates error ─────────────────────────────────────
    def test_stage1_error_propagates(self):
        calls = [("", "Erro ao contatar Ollama (http://localhost:11434): timeout")]

        with patch("src.backend.llm_parser._call_ollama", side_effect=calls):
            result = parse_natural_language("Any text")

        assert result.error is not None
        assert "Ollama" in result.error
        assert result.lp_file == ""

    # ── bad JSON from stage 2 → repair succeeds ─────────────────────────────
    def test_repair_recovers_from_bad_json(self):
        good_json = {"lp_lines": ["2", "1", "1 1", "1 1", "1 1 <= 5"]}
        calls = [
            ("stage1 output", None),
            ("NOT A JSON AT ALL", None),      # stage-2 fails
            (json.dumps(good_json), None),   # repair succeeds
        ]

        with patch("src.backend.llm_parser._call_ollama", side_effect=calls):
            result = parse_natural_language("Problem that confuses stage 2")

        assert result.error is None
        assert result.rationale == "stage1 output"
        lp = parse_text(result.lp_file)
        assert lp.n_vars == 2

    # ── both stage 2 and repair fail → error mentions repair ─────────────────
    def test_repair_also_fails(self):
        calls = [
            ("stage1 output", None),
            ("NOT JSON", None),
            ("ALSO NOT JSON", None),
        ]

        with patch("src.backend.llm_parser._call_ollama", side_effect=calls):
            result = parse_natural_language("Problem that breaks everything")

        assert result.error is not None
        assert "reparo" in result.error.lower()
        assert result.rationale == "stage1 output"
        assert result.lp_file == ""

    # ── lp_file round-trip: parse_text must not raise ─────────────────────────
    def test_lp_file_parseable(self):
        stage3 = {
            "lp_lines": [
                "4", "3", "1 1 1 1", "3 5 2 7",
                "1 2 1 0 <= 10",
                "0 1 3 1 <= 12",
                "2 0 1 2 <= 15",
            ]
        }
        calls = _make_mock_calls("(s1)", stage3)

        with patch("src.backend.llm_parser._call_ollama", side_effect=calls):
            result = parse_natural_language("4-var problem")

        assert result.error is None
        lp = parse_text(result.lp_file)   # must not raise
        assert lp.n_vars == 4
        assert lp.n_cons == 3

    # ── JSON wrapped in markdown code fences is still parsed ─────────────────
    def test_markdown_fence_stripped(self):
        raw_json = '```json\n{"lp_lines": ["2", "1", "1 1", "3 2", "1 1 <= 4"]}\n```'
        calls = [
            ("stage1", None),
            (raw_json, None),
        ]

        with patch("src.backend.llm_parser._call_ollama", side_effect=calls):
            result = parse_natural_language("Fenced JSON")

        assert result.error is None
        lp = parse_text(result.lp_file)
        assert lp.n_vars == 2


# ─────────────────────────── INTEGRATION TESTS ───────────────────────────────

def _ollama_available() -> bool:
    try:
        import ollama
        ollama.Client(host="http://localhost:11434").list()
        return True
    except Exception:
        return False


@pytest.mark.integration
@pytest.mark.skipif(not _ollama_available(), reason="Ollama server not reachable")
class TestLLMIntegration:
    """
    Calls the real LLM pipeline.  Assertions are intentionally lenient
    (we only check structure and parsability, not exact numeric values)
    because the LLM is non-deterministic.
    """

    PROBLEMS = [
        (
            "simple_2var",
            "Maximize 5 times x1 plus 4 times x2, subject to: "
            "6x1 plus x2 is at most 6, and x1 plus x2 is at most 4. "
            "Both variables are non-negative.",
            2, 2,
        ),
        (
            "equality_constraint",
            "A factory produces two products A and B. "
            "Profit is 3 per unit of A and 5 per unit of B. "
            "Total production cannot exceed 200 units. "
            "Exactly 80 units of A must be produced. "
            "Maximize profit. Both non-negative.",
            2, 2,
        ),
        (
            "3var_mixed",
            "Maximize 40x1 + 30x2 + 60x3 subject to: "
            "total x1+x2+x3 at most 10000; "
            "x3 must be at least 2000; "
            "x1 must equal x2; "
            "2x1+3x2+x3 at most 15000. All variables non-negative.",
            3, 4,
        ),
    ]

    @pytest.mark.parametrize("name,problem_text,expected_vars,expected_cons", PROBLEMS)
    def test_pipeline_returns_no_error(self, name, problem_text, expected_vars, expected_cons):
        result = parse_natural_language(problem_text)
        assert result.error is None, (
            f"[{name}] Pipeline returned error:\n{result.error}\n"
            f"Rationale:\n{result.rationale}\n"
            f"Raw:\n{result.raw_response}"
        )

    @pytest.mark.parametrize("name,problem_text,expected_vars,expected_cons", PROBLEMS)
    def test_lp_file_parseable(self, name, problem_text, expected_vars, expected_cons):
        result = parse_natural_language(problem_text)
        if result.error:
            pytest.skip(f"[{name}] Pipeline error (see test_pipeline_returns_no_error)")
        lp = parse_text(result.lp_file)
        assert lp.n_vars == expected_vars, f"[{name}] Expected {expected_vars} vars, got {lp.n_vars}"
        assert lp.n_cons == expected_cons, f"[{name}] Expected {expected_cons} cons, got {lp.n_cons}"

    @pytest.mark.parametrize("name,problem_text,expected_vars,expected_cons", PROBLEMS)
    def test_lp_is_solvable(self, name, problem_text, expected_vars, expected_cons):
        from src.backend.suplex import solve
        result = parse_natural_language(problem_text)
        if result.error:
            pytest.skip("Pipeline error")
        lp  = parse_text(result.lp_file)
        fpi = to_fpi(lp) if False else __import__(
            "src.backend.utils", fromlist=["to_fpi"]
        ).to_fpi(lp)
        sr  = solve(fpi)
        assert sr.status in ("otimo", "inviavel", "ilimitado"), \
            f"[{name}] Unexpected solver status: {sr.status}"
