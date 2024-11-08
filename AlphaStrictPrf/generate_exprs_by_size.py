from itertools import product

from AlphaStrictPrf.strict_prf import C, Expr, P, R, S, Z, expr_to_str_rec


def generate_exprs_by_size(
    n: int, max_p_arity: int, max_c_args: int
) -> list[list[list[Expr]]]:
    if n == 0:
        return [[]]
    if n == 1:
        exprs: list[list[list[Expr]]] = [
            [],
            [[] for _ in range(max_p_arity + 1)],
        ]
        exprs[1][1].extend([Z(), S()])  # S の式をアリティ 1 のリストに追加
        for arity in range(1, max_p_arity + 1):
            exprs[1][arity].extend([P(arity, i) for i in range(1, arity + 1)])
        return exprs
    pre_exprs = generate_exprs_by_size(n - 1, max_p_arity, max_c_args)
    pre_exprs.append([[] for _ in range(max_p_arity + 1)])
    for k in range(1, n - 1):
        for b_arity in range(1, max_p_arity + 1):
            for arity in range(1, max_p_arity + 1):
                pre_exprs[n][arity].extend(
                    [
                        C(base, *args)
                        for base in pre_exprs[k][b_arity]
                        for args in product(
                            pre_exprs[n - k - 1][0]
                            + pre_exprs[n - k - 1][arity],
                            repeat=b_arity,
                        )
                    ]
                )

    for k in range(1, n):
        for s_arity in range(2, max_p_arity):
            pre_exprs[n][s_arity - 1].extend(
                [
                    R(base, arg)
                    for base in pre_exprs[k][0]
                    for arg in pre_exprs[n - k - 1][s_arity]
                ]
            )

    for k in range(1, n):
        for b_arity in range(1, max_p_arity - 1):
            pre_exprs[n][b_arity + 1].extend(
                [
                    R(base, arg)
                    for base in pre_exprs[k][b_arity]
                    for arg in pre_exprs[n - k - 1][b_arity + 2]
                ]
            )

    return pre_exprs


if __name__ == "__main__":
    exprs = generate_exprs_by_size(5, max_p_arity=2, max_c_args=2)
    print(expr_to_str_rec(exprs))
