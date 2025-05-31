def init_exprs(
    max_p_arity: int, eq_domain: list[npt.NDArray]
) -> tuple[list[list[Expr]], list[set[bytes]]]:
    exprs: list[list[Expr]] = [[] for _ in range(max_p_arity + 1)]
    outputs: list[set[bytes]] = [set() for _ in range(max_p_arity + 1)]
    exprs[0].append(Z())
    for input_size in range(1, max_p_arity + 1):
        outputs[input_size].add(output_bytes_const(Z(), input_size, eq_domain))
    try:
        outputs[1].add(output_bytes_not_const(S(), eq_domain))
        exprs[1].append(S())
    except OverflowError:
        pass
    for i in range(1, max_p_arity + 1):
        for j in range(1, i + 1):
            try:
                exprs[i].append(P(i, j))
                outputs[i].add(output_bytes_not_const(P(i, j), eq_domain))
            except OverflowError:
                pass
    return exprs, outputs
