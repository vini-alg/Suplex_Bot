"""
tests/test_solver.py – Unit tests for the Simplex solver (utils + suplex).

Covers:
  - Small  (2 vars, 2 constraints)  – optimal
  - Medium (3 vars, 4 constraints)  – optimal
  - Large  (6 vars, 8 constraints)  – optimal
  - Infeasible example
  - Unbounded example

Run with:
    python -m pytest tests/test_solver.py -v
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.backend.utils import parse_text, to_fpi
from src.backend.suplex import solve


# ─────────────────────────── helpers ─────────────────────────────────────────

def _solve_text(txt: str, policy: str = "largest"):
    lp  = parse_text(txt)
    fpi = to_fpi(lp)
    return solve(fpi, policy=policy)


def _lp_text(*lines):
    return "\n".join(lines)


# ─────────────────────────── SMALL (2 vars, 2 constraints) ───────────────────

class TestSmall:
    """
    MAX  5x1 + 4x2
    s.t. 6x1 +  x2 <= 6
          x1 +  x2 <= 4
    x1, x2 >= 0

    Optimal: x1=2/5, x2=18/5  →  Z = 22  (fractional corner)
    Actually: x* = (0.4, 3.6) => Z = 5*0.4 + 4*3.6 = 2 + 14.4 = 16.4
    Let's verify: the classic 2-var textbook example:
      corner (0,4): Z=16; (1,0): Z=5; corner (2/5, 18/5): Z=5*(2/5)+4*(18/5)=2+14.4=16.4
    """
    TXT = _lp_text("2", "2", "1 1", "5 4", "6 1 <= 6", "1 1 <= 4")

    def test_status_optimal(self):
        r = _solve_text(self.TXT)
        assert r.status == "otimo"

    def test_objective_value(self):
        r = _solve_text(self.TXT)
        assert r.z == pytest.approx(16.4, abs=1e-4)

    def test_primal_feasibility(self):
        r = _solve_text(self.TXT)
        lp = parse_text(self.TXT)
        x = r.x_primal[:lp.n_vars]
        # 6x1 + x2 <= 6
        assert 6*x[0] + x[1] <= 6 + 1e-6
        # x1 + x2 <= 4
        assert x[0] + x[1] <= 4 + 1e-6
        # non-negativity
        assert all(xi >= -1e-6 for xi in x)

    def test_dual_non_negative(self):
        r = _solve_text(self.TXT)
        assert all(yi >= -1e-6 for yi in r.y_dual)

    @pytest.mark.parametrize("policy", ["largest", "bland", "smallest"])
    def test_all_pivot_rules_agree(self, policy):
        r = _solve_text(self.TXT, policy=policy)
        assert r.status == "otimo"
        assert r.z == pytest.approx(16.4, abs=1e-4)


# ─────────────────────────── MEDIUM (3 vars, 4 constraints) ──────────────────

class TestMedium:
    """
    MAX  40x1 + 30x2 + 60x3
    s.t.  x1 +  x2 +  x3 <= 10000
                       x3 >= 2000
         -x1 +  x2        == 0       (x1 = x2)
         2x1 + 3x2 +  x3  <= 15000
    x1, x2, x3 >= 0

    With x1=x2 let k=x1=x2:
      2k+x3 <= 10000; x3 >= 2000; 5k+x3 <= 15000
    Maximise 70k + 60x3.  Since x3 has a higher unit gain, push k→0:
      k=0, x3=10000 → all constraints satisfied, Z = 60*10000 = 600000.
    """
    TXT = _lp_text(
        "3", "4", "1 1 1", "40 30 60",
        "1 1 1 <= 10000",
        "0 0 1 >= 2000",
        "-1 1 0 == 0",
        "2 3 1 <= 15000",
    )

    def test_status_optimal(self):
        r = _solve_text(self.TXT)
        assert r.status == "otimo"

    def test_objective_value(self):
        r = _solve_text(self.TXT)
        assert r.z == pytest.approx(600000.0, abs=1e-2)

    def test_equality_constraint_satisfied(self):
        r = _solve_text(self.TXT)
        lp = parse_text(self.TXT)
        x = r.x_primal[:lp.n_vars]
        assert x[0] == pytest.approx(x[1], abs=1e-4)  # x1 == x2

    def test_x3_lower_bound(self):
        r = _solve_text(self.TXT)
        lp = parse_text(self.TXT)
        x = r.x_primal[:lp.n_vars]
        assert x[2] >= 2000 - 1e-4

    def test_primal_feasibility(self):
        r = _solve_text(self.TXT)
        lp  = parse_text(self.TXT)
        fpi = to_fpi(lp)
        x = r.x_primal[:lp.n_vars]
        # resource constraint
        assert x[0]+x[1]+x[2] <= 10000 + 1e-4
        # budget constraint
        assert 2*x[0]+3*x[1]+x[2] <= 15000 + 1e-4


# ─────────────────────────── LARGE (6 vars, 8 constraints) ───────────────────

class TestLarge:
    """
    MAX  10x1 + 6x2 + 4x3 + 8x4 + 7x5 + 5x6
    s.t.  x1 + x2 + x3 + x4 + x5 + x6 <= 100   (capacity)
          x1                            <= 40
               x2                       <= 30
                    x3                  <= 25
                         x4             <= 35
                              x5        <= 45
          x1 + x2 + x3                  <= 60   (group A)
                         x4 + x5 + x6  <= 70   (group B)
    all >= 0
    """
    TXT = _lp_text(
        "6", "8", "1 1 1 1 1 1", "10 6 4 8 7 5",
        "1 1 1 1 1 1 <= 100",
        "1 0 0 0 0 0 <= 40",
        "0 1 0 0 0 0 <= 30",
        "0 0 1 0 0 0 <= 25",
        "0 0 0 1 0 0 <= 35",
        "0 0 0 0 1 0 <= 45",
        "1 1 1 0 0 0 <= 60",
        "0 0 0 1 1 1 <= 70",
    )

    def test_status_optimal(self):
        r = _solve_text(self.TXT)
        assert r.status == "otimo"

    def test_objective_positive(self):
        r = _solve_text(self.TXT)
        assert r.z > 0

    def test_capacity_constraint(self):
        r = _solve_text(self.TXT)
        lp = parse_text(self.TXT)
        x = r.x_primal[:lp.n_vars]
        assert sum(x) <= 100 + 1e-4

    def test_individual_upper_bounds(self):
        r = _solve_text(self.TXT)
        lp = parse_text(self.TXT)
        x = r.x_primal[:lp.n_vars]
        ubs = [40, 30, 25, 35, 45, float("inf")]
        for i, ub in enumerate(ubs[:5]):
            assert x[i] <= ub + 1e-4

    def test_non_negativity(self):
        r = _solve_text(self.TXT)
        lp = parse_text(self.TXT)
        x = r.x_primal[:lp.n_vars]
        assert all(xi >= -1e-6 for xi in x)

    def test_dual_length(self):
        r = _solve_text(self.TXT)
        lp = parse_text(self.TXT)
        assert len(r.y_dual) == lp.n_cons


# ─────────────────────────── INFEASIBLE ──────────────────────────────────────

class TestInfeasible:
    """
    MAX  x1 + x2
    s.t. x1 + x2 <= 4
         x1 + x2 >= 6     ← contradicts the first constraint
    x1, x2 >= 0
    """
    TXT = _lp_text("2", "2", "1 1", "1 1", "1 1 <= 4", "1 1 >= 6")

    def test_status_infeasible(self):
        r = _solve_text(self.TXT)
        assert r.status == "inviavel"

    def test_no_solution_values(self):
        r = _solve_text(self.TXT)
        assert r.z is None
        assert r.x_primal is None
        assert r.y_dual is None

    def test_logs_mention_infeasibility(self):
        r = _solve_text(self.TXT)
        combined = "\n".join(r.logs).lower()
        assert "invi" in combined  # "inviável" / "inviavel"


# ─────────────────────────── UNBOUNDED ───────────────────────────────────────

class TestUnbounded:
    """
    MAX  x1 + x2
    s.t. x1 - x2 <= 1   ← x2 can grow freely since no upper bound
    x1, x2 >= 0

    As x2→∞ with x1=0: constraint 0-x2<=1 is violated, so try:
    x1 >= 0, x2 >= 0, x1-x2 <= 1: fix x1=1+x2; Z = 1+2*x2 → ∞
    """
    TXT = _lp_text("2", "1", "1 1", "1 2", "1 -1 <= 1")

    def test_status_unbounded(self):
        r = _solve_text(self.TXT)
        assert r.status == "ilimitado"

    def test_no_finite_solution(self):
        r = _solve_text(self.TXT)
        assert r.z is None
        assert r.x_primal is None

    def test_logs_mention_unbounded(self):
        r = _solve_text(self.TXT)
        combined = "\n".join(r.logs).lower()
        assert "ilimitad" in combined
