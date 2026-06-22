#!/usr/bin/env python3
"""
simplex_solver.py – CLI entry point for the Suplex Simplex solver.

Usage examples:
  python simplex_solver.py examples/example_2var.txt
  python simplex_solver.py examples/example_phase1.txt --policy bland --decimals 6
  python simplex_solver.py examples/example_3var.txt --digits 12 --decimals 3
"""

import sys
import argparse
from pathlib import Path

# Allow running from repo root without installing as package
sys.path.insert(0, str(Path(__file__).parent))

from src.backend.utils import parse_file, to_fpi
from src.backend.suplex import solve


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="simplex_solver",
        description="Suplex – Solver do Método Simplex (Fase I + Fase II)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Formato do arquivo de entrada (TXT):
  Linha 1 : número de variáveis
  Linha 2 : número de restrições
  Linha 3 : domínio (1=x≥0  -1=x≤0  0=livre), separado por espaços
  Linha 4 : coeficientes da função objetivo (MAXIMIZAÇÃO)
  Linhas 5+: <coeficientes> <= / >= / == <RHS>

Exemplo:
  2
  2
  1 1
  5 4
  6 1 <= 6
  1 1 <= 4
        """,
    )
    p.add_argument("filename", help="Caminho para o arquivo de entrada .txt")
    p.add_argument(
        "--decimals", type=int, default=4, metavar="N",
        help="Casas decimais para exibição dos valores (padrão: 4)",
    )
    p.add_argument(
        "--digits", type=int, default=10, metavar="N",
        help="Largura de coluna no tableau (padrão: 10)",
    )
    p.add_argument(
        "--policy", choices=["largest", "bland", "smallest"], default="largest",
        help="Regra de entrada da variável na base (padrão: largest)",
    )
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    path = Path(args.filename)
    if not path.exists():
        parser.error(f"Arquivo não encontrado: {path}")

    print(f"\n{'='*60}")
    print(f"  Suplex Bot – Simplex Solver")
    print(f"  Arquivo : {path}")
    print(f"  Política: {args.policy}  |  Decimais: {args.decimals}  |  Colunas: {args.digits}")
    print(f"{'='*60}\n")

    # ── Parse & preprocess ────────────────────────────────────────────────────
    lp = parse_file(str(path))
    print(f"Problema lido: {lp.n_vars} variável(is), {lp.n_cons} restrição(ões).")
    print(f"Objetivo (max): {lp.c}")
    print(f"Domínio: {lp.domain}\n")

    fpi = to_fpi(lp)
    if fpi.logs:
        print("── Pré-processamento (FPI) ──")
        for msg in fpi.logs:
            print(" ·", msg)
        print()

    # ── Solve ─────────────────────────────────────────────────────────────────
    def _cb(msg: str):
        print(msg)

    result = solve(
        fpi,
        policy=args.policy,
        decimals=args.decimals,
        digits=args.digits,
        log_callback=_cb,
    )

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  STATUS FINAL : {result.status.upper()}")
    print(f"{'='*60}")

    if result.status == "otimo":
        print(f"  Z ótimo      : {result.z:.{args.decimals}f}")
        print("\n  Solução Primal:")
        for j, name in enumerate(result.var_names):
            print(f"    {name:>8} = {result.x_primal[j]:.{args.decimals}f}")
        print("\n  Solução Dual (preços-sombra):")
        for i, yi in enumerate(result.y_dual):
            print(f"    y{i+1:>6} = {yi:.{args.decimals}f}")
        print(f"\n  Iterações    : {result.iterations}")

    elif result.status == "inviavel":
        print("  O problema não possui solução viável.")

    elif result.status == "ilimitado":
        print("  O problema é ilimitado (Z → ∞).")

    print(f"{'='*60}\n")
    return 0 if result.status == "otimo" else 1


if __name__ == "__main__":
    sys.exit(main())
