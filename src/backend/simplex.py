"""
simplex.py – Simplex Method solver (Phase I + Phase II) from scratch.

Tableau layout  (each row is a numpy 1-D slice):
  rows 0..m-1  : constraint rows  [A | b]
  row  m       : objective row    [-c | z]   (stored as reduced costs; maximisation)

Pivoting is done entirely with numpy slicing – NO scipy or linalg solvers.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Tuple

from .utils import FPI, format_tableau

EPS = 1e-9   # numerical zero tolerance


# ─────────────────────────── Result dataclass ────────────────────────────────

@dataclass
class SolveResult:
    status: str                   # 'otimo' | 'otimo_multiplos' | 'inviavel' | 'ilimitado'
    z: Optional[float]
    x_primal: Optional[np.ndarray]
    y_dual: Optional[np.ndarray]
    var_names: List[str]
    iterations: int
    tableau_history: List[np.ndarray] = field(default_factory=list)
    basis_history: List[List[int]] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)
    alternate_solutions: List[np.ndarray] = field(default_factory=list)


# ─────────────────────────── Pivot selection rules ───────────────────────────

def _entering_largest(obj_row: np.ndarray, n_cols: int) -> int:
    """Most-positive reduced cost (largest coefficient rule)."""
    costs = obj_row[:n_cols]
    idx = int(np.argmax(costs))
    return idx if costs[idx] > EPS else -1


def _entering_bland(obj_row: np.ndarray, n_cols: int) -> int:
    """Bland's rule: smallest index with positive reduced cost."""
    for j in range(n_cols):
        if obj_row[j] > EPS:
            return j
    return -1


def _entering_smallest(obj_row: np.ndarray, n_cols: int) -> int:
    """Smallest positive reduced cost (least improvement)."""
    costs = obj_row[:n_cols]
    pos = [(costs[j], j) for j in range(n_cols) if costs[j] > EPS]
    if not pos:
        return -1
    return min(pos)[1]


_RULES = {
    "largest": _entering_largest,
    "bland":   _entering_bland,
    "smallest":_entering_smallest,
}


# ─────────────────────────── Ratio test (leaving variable) ───────────────────

def _leaving(tableau: np.ndarray, enter: int, m: int) -> int:
    """
    Minimum-ratio test. Returns row index of leaving variable.
    Returns -1 if the problem is unbounded.
    """
    ratios = []
    for i in range(m):
        aij = tableau[i, enter]
        if aij > EPS:
            ratios.append((tableau[i, -1] / aij, i))
    if not ratios:
        return -1
    return min(ratios)[1]


# ─────────────────────────── Pivot operation ─────────────────────────────────

def _pivot(tableau: np.ndarray, row: int, col: int) -> None:
    """In-place Gauss-Jordan pivot on (row, col)."""
    pivot_val = tableau[row, col]
    tableau[row, :] /= pivot_val
    for i in range(tableau.shape[0]):
        if i != row:
            tableau[i, :] -= tableau[i, col] * tableau[row, :]


# ─────────────────────────── Simplex iterations ──────────────────────────────

def _run_simplex(
    tableau: np.ndarray,
    basis: List[int],
    var_names: List[str],
    policy: str,
    logs: List[str],
    tableau_history: List[np.ndarray],
    basis_history: List[List[int]],
    decimals: int,
    digits: int,
    max_iter: int = 500,
    log_callback: Optional[Callable[[str], None]] = None,
) -> str:
    """
    Core simplex loop. Mutates `tableau` and `basis` in-place.
    Appends to `logs`, `tableau_history`, `basis_history`.
    Returns 'otimo' | 'ilimitado'.
    """
    m = tableau.shape[0] - 1   # constraint rows
    n_cols = tableau.shape[1] - 1
    enter_fn = _RULES[policy]

    def _log(msg: str):
        logs.append(msg)
        if log_callback:
            log_callback(msg)

    tableau_history.append(tableau.copy())
    basis_history.append(list(basis))

    for it in range(max_iter):
        enter = enter_fn(tableau[-1], n_cols)
        if enter == -1:
            _log("Nenhum custo reduzido positivo. Solução ótima encontrada.")
            return "otimo"

        leave_row = _leaving(tableau, enter, m)
        if leave_row == -1:
            _log(f"Coluna {var_names[enter] if enter < len(var_names) else enter} ilimitada. Problema ILIMITADO.")
            return "ilimitado"

        leave_var = basis[leave_row]
        ename = var_names[enter] if enter < len(var_names) else f"col{enter}"
        lname = var_names[leave_var] if leave_var < len(var_names) else f"col{leave_var}"
        _log(
            f"[Iter {it+1}] Política '{policy}': entra {ename} (custo reduzido "
            f"{tableau[-1, enter]:.{decimals}f}), sai {lname} (linha {leave_row+1})."
        )

        _pivot(tableau, leave_row, enter)
        basis[leave_row] = enter

        tableau_history.append(tableau.copy())
        basis_history.append(list(basis))

        t_str = format_tableau(tableau, var_names, basis, decimals, digits)
        _log(f"Tableau após iteração {it+1}:\n{t_str}")

    _log("Limite de iterações atingido.")
    return "limite"


# ─────────────────────────── Phase I ─────────────────────────────────────────

def phase_one(
    fpi: FPI,
    logs: List[str],
    tableau_history: List[np.ndarray],
    basis_history: List[List[int]],
    decimals: int,
    digits: int,
    log_callback: Optional[Callable[[str], None]] = None,
) -> Tuple[Optional[np.ndarray], Optional[List[int]], Optional[List[str]]]:
    """
    Build and solve the auxiliary LP to find an initial feasible basis.
    Returns (tableau_fpi, basis, var_names_fpi) or (None, None, None) if infeasible.

    Artificial variables are added for constraints that do NOT already have
    a slack variable acting as a unit basis column (i.e., >= and == rows).
    """
    m = fpi.A_fpi.shape[0]
    n_fpi = fpi.A_fpi.shape[1]

    # Determine which rows need artificials:
    # A row i needs an artificial if it has no ready-made +1 unit column in A_fpi.
    # In practice: rows where the corresponding slack column is -1 or 0.
    slack_start = n_fpi - fpi.n_slack  # index where slack columns begin
    rows_needing_art = []
    for i in range(m):
        slack_col = slack_start + i
        if fpi.A_fpi[i, slack_col] <= 0:
            rows_needing_art.append(i)

    def _log(msg: str):
        logs.append(msg)
        if log_callback:
            log_callback(msg)

    if not rows_needing_art:
        _log("Todas as restrições possuem variável de folga ≥ 0. Fase I desnecessária.")
        # Build initial basis directly from slack columns
        basis = [slack_start + i for i in range(m)]
        tableau = _build_tableau(fpi.c_fpi, fpi.A_fpi, fpi.b_fpi, basis)
        return tableau, basis, list(fpi.var_names)

    _log(
        f"Fase I: adicionando {len(rows_needing_art)} variável(is) artificial(is) "
        f"para as linhas: {[r+1 for r in rows_needing_art]}."
    )

    n_art = len(rows_needing_art)
    art_names = [f"a{i+1}" for i in range(n_art)]
    var_names_aux = list(fpi.var_names) + art_names

    # Build auxiliary A: append identity columns for artificials
    art_cols = np.zeros((m, n_art), dtype=float)
    for k, row_idx in enumerate(rows_needing_art):
        art_cols[row_idx, k] = 1.0

    A_aux = np.hstack([fpi.A_fpi, art_cols])

    # Auxiliary objective: minimise sum of artificials → maximise -sum
    c_aux = np.zeros(n_fpi + n_art, dtype=float)
    for k in range(n_art):
        c_aux[n_fpi + k] = -1.0  # we store as max, so penalise positively

    # Initial basis: slack where available, artificial otherwise
    basis_aux: List[int] = []
    art_ptr = 0
    for i in range(m):
        slack_col = slack_start + i
        if fpi.A_fpi[i, slack_col] > 0:
            basis_aux.append(slack_col)
        else:
            basis_aux.append(n_fpi + art_ptr)
            art_ptr += 1

    tableau_aux = _build_tableau(c_aux, A_aux, fpi.b_fpi, basis_aux)
    _log("Tableau inicial da Fase I:\n" + format_tableau(tableau_aux, var_names_aux, basis_aux, decimals, digits))

    status = _run_simplex(
        tableau_aux, basis_aux, var_names_aux,
        "bland",   # always Bland for Phase I to guarantee termination
        logs, tableau_history, basis_history, decimals, digits,
        log_callback=log_callback,
    )

    # tableau[-1,-1] = -z_aux; z_aux = -sum_of_artificials → sum_of_artificials = tableau[-1,-1]
    sum_of_artificials = tableau_aux[-1, -1]
    z_aux = -sum_of_artificials
    _log(f"Fase I concluída. Soma das variáveis artificiais = {sum_of_artificials:.{decimals}f}.")

    if abs(sum_of_artificials) > 1e-6:
        _log("Problema INVIÁVEL: não foi possível zerar as variáveis artificiais.")
        return None, None, None

    # Check if any artificial is still in basis (degenerate)
    art_indices = set(range(n_fpi, n_fpi + n_art))
    for k, b in enumerate(basis_aux):
        if b in art_indices:
            # Try to pivot it out
            pivoted = False
            for j in range(n_fpi):
                if abs(tableau_aux[k, j]) > EPS and j not in basis_aux:
                    _log(f"Variável artificial {var_names_aux[b]} ainda na base (degenerada); pivoteando para remover.")
                    _pivot(tableau_aux, k, j)
                    basis_aux[k] = j
                    pivoted = True
                    break
            if not pivoted:
                _log(f"Linha {k+1} é redundante (artificial não removível); descartando.")

    # Strip artificial columns and restore original objective
    tableau_fpi = tableau_aux[:, list(range(n_fpi)) + [-1]].copy()  # drop art cols, keep b
    var_names_fpi = list(fpi.var_names)
    # Recompute objective row from scratch
    c_row = _compute_obj_row(fpi.c_fpi, tableau_fpi, basis_aux)
    tableau_fpi[-1, :] = c_row
    _log("Variáveis artificiais removidas. Restaurando função objetivo original para a Fase II.")
    _log("Tableau inicial da Fase II:\n" + format_tableau(tableau_fpi, var_names_fpi, basis_aux, decimals, digits))

    return tableau_fpi, basis_aux, var_names_fpi


# ─────────────────────────── Multiple-optima detection ──────────────────────

def _find_alternate_solutions(
    tableau: np.ndarray,
    basis: List[int],
    n_structural: int,
) -> List[np.ndarray]:
    """
    After reaching optimality, check each non-basic structural variable for
    zero reduced cost.  For each such variable, pivot it in on a copy of the
    tableau to obtain the alternate optimal BFS.
    Returns a (possibly empty) list of full x_primal vectors.
    """
    m = tableau.shape[0] - 1
    n_total = tableau.shape[1] - 1
    basis_set = set(basis)
    alternates: List[np.ndarray] = []

    for j in range(n_structural):
        if j in basis_set or abs(tableau[-1, j]) >= EPS:
            continue
        ratios = [
            (tableau[i, -1] / tableau[i, j], i)
            for i in range(m) if tableau[i, j] > EPS
        ]
        if not ratios:
            continue
        leave_row = min(ratios)[1]
        tab = tableau.copy()
        bas = list(basis)
        _pivot(tab, leave_row, j)
        bas[leave_row] = j
        x = np.zeros(n_total, dtype=float)
        for i, b in enumerate(bas):
            if b < n_total:
                x[b] = tab[i, -1]
        alternates.append(x)

    return alternates


# ─────────────────────────── Tableau builder helpers ─────────────────────────

def _build_tableau(c: np.ndarray, A: np.ndarray, b: np.ndarray, basis: List[int]) -> np.ndarray:
    """
    Construct initial tableau  [A | b ; -c_bar | z_bar]
    where the objective row is already adjusted for the current basis.
    """
    m, n = A.shape
    T = np.zeros((m + 1, n + 1), dtype=float)
    T[:m, :n] = A
    T[:m, -1] = b
    c_row = _compute_obj_row(c, T, basis)
    T[-1, :] = c_row
    return T


def _compute_obj_row(c: np.ndarray, T: np.ndarray, basis: List[int]) -> np.ndarray:
    """
    Compute reduced-cost objective row for maximisation.
    Stored as (c_j - z_j) but with sign inverted for internal accounting:
      T[-1, j] = c_j - z_j  (positive → can improve)
      T[-1, -1] = current z  (negated: we store -z so that max(-z) = min z ... actually we store z directly)

    Convention used here:
      T[-1, j] = reduced cost  c̄_j  (positive means entering)
      T[-1,-1] = -z_bar            (we negate so the objective row value at b column is -z)
    Actually, let's use the standard convention:
      objective row stores [-c̄_j] but updated so that the 'b' entry is z.
    We'll use: obj row = (c - c_B @ B^{-1} @ A) and z = c_B @ x_B = c_B @ B^{-1} @ b.
    Since we already have B^{-1} A in T (after pivots), we just do one full update.
    """
    m = T.shape[0] - 1
    n = T.shape[1] - 1  # includes b column

    # c_bar_j = c_j - c_B @ (column j of B^{-1}A)
    obj = np.zeros(n + 1, dtype=float)
    for j in range(n):
        if j < len(c):
            cj = c[j]
        else:
            cj = 0.0
        obj[j] = cj - sum(
            (c[basis[i]] if basis[i] < len(c) else 0.0) * T[i, j]
            for i in range(m)
        )
    # Store -z so Gauss-Jordan pivot on the obj row increases z correctly
    obj[-1] = -sum(
        (c[basis[i]] if basis[i] < len(c) else 0.0) * T[i, -1]
        for i in range(m)
    )
    return obj


# ─────────────────────────── Main entry point ────────────────────────────────

def solve(
    fpi: FPI,
    policy: str = "largest",
    decimals: int = 4,
    digits: int = 10,
    log_callback: Optional[Callable[[str], None]] = None,
) -> SolveResult:
    """
    Solve the FPI problem using Phase I + Phase II Simplex.

    Parameters
    ----------
    fpi      : FPI object from utils.to_fpi()
    policy   : pivot rule ('largest' | 'bland' | 'smallest')
    decimals : decimal places for display
    digits   : column width for tableau display
    log_callback : optional function(msg) called for each log message

    Returns
    -------
    SolveResult
    """
    logs: List[str] = list(fpi.logs)
    t_history: List[np.ndarray] = []
    b_history: List[List[int]] = []

    def _log(msg: str):
        logs.append(msg)
        if log_callback:
            log_callback(msg)

    _log("=" * 60)
    _log("INÍCIO DA RESOLUÇÃO DO SIMPLEX")
    _log(f"Política de entrada: {policy} | Decimais: {decimals}")
    _log(f"Variáveis FPI: {fpi.var_names}")
    _log("=" * 60)

    # ── Phase I ──────────────────────────────────────────────────────────────
    tableau, basis, var_names = phase_one(
        fpi, logs, t_history, b_history, decimals, digits, log_callback=log_callback
    )

    if tableau is None:
        return SolveResult(
            status="inviavel", z=None, x_primal=None, y_dual=None,
            var_names=fpi.var_names, iterations=len(t_history),
            tableau_history=t_history, basis_history=b_history, logs=logs,
        )

    # ── Phase II ─────────────────────────────────────────────────────────────
    _log("=" * 60)
    _log("FASE II – Optimização da função objetivo original")
    _log("=" * 60)

    status = _run_simplex(
        tableau, basis, var_names,
        policy, logs, t_history, b_history, decimals, digits,
        log_callback=log_callback,
    )

    if status == "ilimitado":
        return SolveResult(
            status="ilimitado", z=None, x_primal=None, y_dual=None,
            var_names=var_names, iterations=len(t_history),
            tableau_history=t_history, basis_history=b_history, logs=logs,
        )

    # ── Extract solution ──────────────────────────────────────────────────────
    m = tableau.shape[0] - 1
    n_total = tableau.shape[1] - 1
    z = -tableau[-1, -1]   # RHS stores -z

    x_primal = np.zeros(n_total, dtype=float)
    for i, b in enumerate(basis):
        if b < n_total:
            x_primal[b] = tableau[i, -1]

    # Dual solution: reduced costs of slack variables (shadow prices)
    slack_start = len(fpi.var_names) - fpi.n_slack
    y_dual = np.zeros(m, dtype=float)
    for i in range(m):
        slack_col = slack_start + i
        if slack_col < n_total:
            y_dual[i] = -tableau[-1, slack_col]   # y_i = -c̄_{s_i} for <= constraints

    # ── Detect multiple optimal solutions ────────────────────────────────────
    n_structural = fpi.n_original + fpi.n_free_aux
    alternates = _find_alternate_solutions(tableau, basis, n_structural)
    final_status = "otimo_multiplos" if alternates else "otimo"

    _log("=" * 60)
    _log(f"STATUS: {'ÓTIMO (MÚLTIPLOS)' if alternates else 'ÓTIMO'}")
    _log(f"Valor ótimo Z = {z:.{decimals}f}")
    _log("Solução Primal:")
    for j, name in enumerate(var_names):
        _log(f"  {name} = {x_primal[j]:.{decimals}f}")
    if alternates:
        _log(f"Soluções alternativas ({len(alternates)} encontrada(s)):")
        for k, alt in enumerate(alternates):
            _log(f"  Alt {k+1}: " + "  ".join(f"{name}={alt[j]:.{decimals}f}" for j, name in enumerate(var_names)))
    _log("Solução Dual (preços-sombra):")
    for i, yi in enumerate(y_dual):
        _log(f"  y{i+1} = {yi:.{decimals}f}")
    _log("=" * 60)

    return SolveResult(
        status=final_status,
        z=z,
        x_primal=x_primal,
        y_dual=y_dual,
        var_names=var_names,
        iterations=len(t_history) - 1,
        tableau_history=t_history,
        basis_history=b_history,
        logs=logs,
        alternate_solutions=alternates,
    )
