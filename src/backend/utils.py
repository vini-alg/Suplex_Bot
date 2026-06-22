"""
utils.py – Input file parser and preprocessing for the Simplex solver.

File format (TXT):
  Line 1 : number of original variables (n)
  Line 2 : number of constraints (m)
  Line 3 : domain flags, space-separated  (1 = x>=0, -1 = x<=0, 0 = free)
  Line 4 : objective coefficients (maximisation), space-separated
  Lines 5+: constraint rows  <coefs>  <sign>  <rhs>
             sign is one of:  <=  >=  ==
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class LP:
    """Canonical representation BEFORE FPI conversion."""
    n_vars: int
    n_cons: int
    domain: List[int]          # 1 | -1 | 0  per original variable
    c: np.ndarray              # objective (maximise)
    A: np.ndarray              # constraint matrix (m x n)
    b: np.ndarray              # RHS
    signs: List[str]           # '<=' | '>=' | '=='


@dataclass
class FPI:
    """Problem in Standard Form (Forma Padrão / FPI)."""
    c_fpi: np.ndarray          # objective in FPI (maximise)
    A_fpi: np.ndarray          # constraint matrix (m x total_vars)
    b_fpi: np.ndarray          # RHS (all >= 0 after row sign flip)
    n_original: int            # number of original variables
    n_slack: int               # slack/surplus variables added
    n_free_aux: int            # extra columns for free variable splits
    var_names: List[str]       # human-readable names for all columns
    logs: List[str] = field(default_factory=list)


def parse_file(path: str) -> LP:
    """Read the structured TXT file and return an LP dataclass."""
    with open(path, "r", encoding="utf-8") as f:
        raw = [ln.strip() for ln in f if ln.strip()]

    n = int(raw[0])
    m = int(raw[1])
    domain = list(map(int, raw[2].split()))
    c = np.array(list(map(float, raw[3].split())), dtype=float)

    if len(domain) != n:
        raise ValueError(f"Domain line must have {n} entries, got {len(domain)}.")
    if len(c) != n:
        raise ValueError(f"Objective line must have {n} entries, got {len(c)}.")

    A_rows, b_rows, signs = [], [], []
    for i, line in enumerate(raw[4: 4 + m]):
        parts = line.split()
        sign = parts[-2]
        rhs = float(parts[-1])
        coefs = list(map(float, parts[:-2]))
        if sign not in ("<=", ">=", "=="):
            raise ValueError(f"Constraint {i+1}: unknown sign '{sign}'.")
        if len(coefs) != n:
            raise ValueError(f"Constraint {i+1}: expected {n} coefs, got {len(coefs)}.")
        A_rows.append(coefs)
        b_rows.append(rhs)
        signs.append(sign)

    return LP(
        n_vars=n,
        n_cons=m,
        domain=domain,
        c=c,
        A=np.array(A_rows, dtype=float),
        b=np.array(b_rows, dtype=float),
        signs=signs,
    )


def parse_text(text: str) -> LP:
    """Same as parse_file but accepts a raw string (for Streamlit use)."""
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as tf:
        tf.write(text)
        tmp = tf.name
    try:
        return parse_file(tmp)
    finally:
        os.unlink(tmp)


def to_fpi(lp: LP) -> FPI:
    """
    Convert an LP to standard form (FPI – Forma Padrão com Igualdades).
    Steps:
      1. Handle domain: x<=0 → substitute x'=-x; free → split x = x+ - x-
      2. Ensure b >= 0 by flipping row signs where needed.
      3. Convert inequalities to equalities via slack/surplus variables.
    Returns an FPI object with transformation logs.
    """
    logs: List[str] = []
    n, m = lp.n_vars, lp.n_cons
    A = lp.A.copy()
    b = lp.b.copy()
    c = lp.c.copy()
    signs = list(lp.signs)

    # ── Step 1: variable substitutions ──────────────────────────────────────
    # Build mapping: original index → list of FPI column indices + transform
    col_names: List[str] = []
    c_new_cols: List[float] = []
    A_new_cols: List[np.ndarray] = []

    for j in range(n):
        d = lp.domain[j]
        if d == 1:
            col_names.append(f"x{j+1}")
            c_new_cols.append(c[j])
            A_new_cols.append(A[:, j].copy())
        elif d == -1:
            logs.append(f"Variável x{j+1} ≤ 0: substituindo por x{j+1}' = -x{j+1} (x{j+1}' ≥ 0).")
            col_names.append(f"x{j+1}'")
            c_new_cols.append(-c[j])
            A_new_cols.append(-A[:, j])
        else:  # free
            logs.append(
                f"Variável x{j+1} livre: dividindo em x{j+1}+ - x{j+1}- (ambas ≥ 0)."
            )
            col_names.append(f"x{j+1}+")
            col_names.append(f"x{j+1}-")
            c_new_cols.extend([c[j], -c[j]])
            A_new_cols.append(A[:, j].copy())
            A_new_cols.append(-A[:, j])

    n_free_aux = len(col_names) - n  # extra columns from free splits
    A_sub = np.column_stack(A_new_cols)  # (m x n_fpi_orig)
    c_sub = np.array(c_new_cols, dtype=float)

    # ── Step 2: ensure b >= 0 ────────────────────────────────────────────────
    for i in range(m):
        if b[i] < 0:
            logs.append(
                f"Restrição {i+1}: RHS negativo ({b[i]}), multiplicando a linha por -1 e invertendo o sinal."
            )
            A_sub[i, :] *= -1
            b[i] *= -1
            if signs[i] == "<=":
                signs[i] = ">="
            elif signs[i] == ">=":
                signs[i] = "<="

    # ── Step 3: add slack/surplus variables ──────────────────────────────────
    slack_cols: List[np.ndarray] = []
    slack_names: List[str] = []
    slack_c: List[float] = []
    slack_idx = len(col_names)

    for i in range(m):
        s_name = f"s{i+1}"
        col = np.zeros(m, dtype=float)
        if signs[i] == "<=":
            logs.append(f"Restrição {i+1} (≤): adicionando variável de folga {s_name}.")
            col[i] = 1.0
        elif signs[i] == ">=":
            logs.append(f"Restrição {i+1} (≥): subtraindo variável de excesso {s_name}.")
            col[i] = -1.0
        else:  # ==
            logs.append(f"Restrição {i+1} (=): sem variável de folga/excesso.")
            # zero column – artificial will be added in Phase I
        slack_cols.append(col)
        slack_names.append(s_name)
        slack_c.append(0.0)

    n_slack = m
    A_slack = np.column_stack(slack_cols)  # (m x m)

    A_fpi = np.hstack([A_sub, A_slack])
    c_fpi = np.concatenate([c_sub, slack_c])
    var_names = col_names + slack_names

    return FPI(
        c_fpi=c_fpi,
        A_fpi=A_fpi,
        b_fpi=b,
        n_original=n,
        n_slack=n_slack,
        n_free_aux=n_free_aux,
        var_names=var_names,
        logs=logs,
    )


def format_tableau(tableau: np.ndarray, var_names: List[str], basis: List[int],
                   decimals: int = 4, digits: int = 10) -> str:
    """Return a pretty-printed string of a simplex tableau."""
    m, total = tableau.shape
    n_cols = total - 1  # last column is b
    col_w = max(digits, 6)

    header_names = [var_names[j] if j < len(var_names) else f"a{j}" for j in range(n_cols)] + ["b"]
    header = "  Basis  |" + "".join(f"{h:>{col_w}}" for h in header_names)
    sep = "-" * len(header)

    lines = [sep, header, sep]
    for i in range(m - 1):  # skip last row (objective)
        b_name = var_names[basis[i]] if basis[i] < len(var_names) else f"a{basis[i]}"
        row = f"  {b_name:>6} |"
        for val in tableau[i]:
            row += f"{val:>{col_w}.{decimals}f}"
        lines.append(row)

    lines.append(sep)
    obj_row = "  z      |"
    for val in tableau[-1]:
        obj_row += f"{val:>{col_w}.{decimals}f}"
    lines.append(obj_row)
    lines.append(sep)
    return "\n".join(lines)
