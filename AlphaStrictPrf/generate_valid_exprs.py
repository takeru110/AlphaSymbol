from itertools import combinations
from typing import Dict, List

import pandas as pd
from strict_prf import C, Expr, P, R, S, Z


def generate_valid_exprs(
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
        expressions_by_arity.setdefault(n, []).extend(P(n, i) for i in range(1, n + 1))

    if depth > 1:
        smaller_expressions_by_arity = generate_expressions(
            depth - 1, max_p_arity, max_c_args
        )

        # Generate composite functions C with different number of arguments
        for func_arity, func_exprs in smaller_expressions_by_arity.items():
            # Expressions used for args
            if func_arity is None:
                continue
            for args_arity, args_exprs in smaller_expressions_by_arity.items():
                """
                
                CはNoneの関数も含んでいいことを反映しなけれない
                
                """
                args_exprs_not_none = [expr for expr in args_exprs if expr is not None]
                new_c_exprs = [
                    C(func_expr, *args)
                    for args in combinations(args_exprs_not_none, args_arity)
                    for func_expr in func_exprs
                ]
                expressions_by_arity.setdefault(args_arity, []).extend(new_c_exprs)

        # Generate recursive functions R(Expr1, Expr2)
        for base_arity, base_exprs in smaller_expressions_by_arity.items():
            if base_arity is None:
                # If base has None arity, allow steps with arity >= 2
                for (
                    step_arity,
                    step_exprs,
                ) in smaller_expressions_by_arity.items():
                    if step_arity is not None and step_arity >= 2:
                        new_r_exprs = [
                            R(base_expr, step_expr)
                            for base_expr in base_exprs
                            for step_expr in step_exprs
                        ]
                        expressions_by_arity.setdefault(step_arity - 1, []).extend(
                            new_r_exprs
                        )
            else:
                # If base has a defined arity, step must have arity = base_arity + 2
                step_arity = base_arity + 2
                if step_arity in smaller_expressions_by_arity:
                    new_r_exprs = [
                        R(base_expr, step_expr)
                        for base_expr in base_exprs
                        for step_expr in smaller_expressions_by_arity[step_arity]
                    ]
                    expressions_by_arity.setdefault(base_arity + 1, []).extend(
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
