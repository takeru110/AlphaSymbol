from itertools import combinations
from typing import Dict, List

import pandas as pd
from strict_prf import C, Expr, P, R, S, Z


def generate_expressions(
    depth: int,
    max_p_arity: int,
    max_c_args: int,
) -> Dict[int, List[Expr]]:
    """
    Generate all possible PRF expressions up to a given depth.

    Args:
    - depth (int): The depth of recursion for generating expressions.
    - max_c_args (int): Maximum number of arguments allowed for C.

    Returns:
    - Dict[int, List[Expr]]: Dictionary of generated PRF expressions grouped by their arity.
    """
    # Base case: depth == 1, return the simplest expressions (Z, S, P)
    expressions_by_arity = {
        None: [Z()],
        1: [S()],
    }  # Base zero and successor functions

    # Add projection functions for the input arity
    for n in range(1, max_p_arity + 1):
        expressions_by_arity.setdefault(n, []).extend(
            P(n, i) for i in range(1, n + 1)
        )

    if depth > 1:
        smaller_expressions_by_arity = generate_expressions(
            depth - 1, max_p_arity, max_c_args
        )

        # Generate composite functions C with different number of arguments
        for num_args in range(1, max_c_args):
            # Expressions used for args
            for args_arity, exprs in smaller_expressions_by_arity.items():
                valid_funcs = smaller_expressions_by_arity.get(num_args, [])
                if args_arity is None:
                    args = (Z() for _ in range(num_args))
                    new_c_exprs = [C(func, *args) for func in valid_funcs]
                else:
                    new_c_exprs = [
                        C(func, *args)
                        for args in combinations(exprs, num_args)
                        for func in valid_funcs
                    ]
                expressions_by_arity.setdefault(args_arity, []).extend(
                    new_c_exprs
                )

        # Generate recursive functions R(Expr1, Expr2)
        for arity1, exprs1 in smaller_expressions_by_arity.items():
            if arity1 is None:
                for arity2, exprs2 in smaller_expressions_by_arity.items():
                    if arity2 is not None and arity2 <= 2:
                        new_r_exprs = [
                            R(expr1, expr2)
                            for expr1 in exprs1
                            for expr2 in exprs2
                        ]
                        expressions_by_arity.setdefault(arity2 - 1, []).extend(
                            new_r_exprs
                        )
            else:
                arity2 = arity1 + 2
                exprs2 = smaller_expressions_by_arity.get(arity2, [])
                new_r_exprs = [
                    R(expr1, expr2) for expr1 in exprs1 for expr2 in exprs2
                ]
                expressions_by_arity.setdefault(arity1 + 1, []).extend(
                    new_r_exprs
                )

    return expressions_by_arity


# Example usage
expressions = generate_expressions(depth=3, max_p_arity=3, max_c_args=3)

expr_li = []
for arity, exprs in expressions.items():
    print(f"Arity {arity}: {len(exprs)} expressions")
    for expr in exprs:
        expr_li.append(
            {
                "Arity": arity,
                "Expressions": str(expr),
            }
        )


df = pd.DataFrame(expr_li)
df.to_csv("valid_expressions.csv", index=False)

print("Done")
