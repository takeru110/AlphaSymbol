from collections import deque
from typing import Any, Deque, List

MAX_EVALUATION_LIMIT = 10**2


class InputSizeError(Exception):
    "the number of inputs is invalid for the arity of then function"

    pass


class OverflowLimitExceededError(Exception):
    """
    a value exceeds the overflow limit in the evaluation of the function
    """

    pass


class Expr:
    def __eq__(self, other):
        """Expr型のクラスの等価性を確認する"""
        if type(self) is not type(other):
            return False
        return self._eq_impl(other)

    def __str__(self):
        return NotImplementedError()

    def _eq_impl(self, other):
        """派生クラスで実装する等価性の比較"""
        raise NotImplementedError()

    def __hash__(self):
        """Expr型のクラスのハッシュ値を計算する"""
        return hash(self._hash_impl())

    def _hash_impl(self):
        """派生クラスで実装するハッシュ値の計算"""
        raise NotImplementedError()

    def evaluate(self, *args: int) -> int:
        """
        PRFが意味する自然数関数に変換して入力に対する値を評価する

        Args:
        - args (int): 自然数関数に変換したPRFの引数に対応する自然数のリスト

        Returns:
        - int: PRFを自然数関数に変換したときの入力に対する出力値


        Raises:
        - InputSizeError: 入力リストのサイズとPRFのarityがが合わないとき
        - OverflowLimitExceededError: 計算結果が設定した上限(再帰の上限)を超えたとき
        """
        raise NotImplementedError()

    def tree_string(self, indent: int = 0) -> str:
        """語の木構造をインデント付きで出力する"""
        raise NotImplementedError()

    def is_valid(self):
        """自然数関数が定義できるかをチェックする

        自然数関数に直したとき、arityの数などで矛盾が生じないかを再帰的に確認する。

        Return:
            Bool: 自然数関数に変換しようとしたとき、引数の数が合わないなどによって変換不可能であるときFalse
        """
        raise NotImplementedError()

    def arity(self):
        """自然数関数にしたときの引数の数
        Return:
            int: 自然数関数に直したときの引数の数
        AssertionError:
            そもそも自然数関数に直せないときに起きる
        """
        raise NotImplementedError()

    def positions(self) -> List[Deque[int]]:
        """
        Generates all possible positions up to a certain depth.

        Args:
            depth (int): The maximum depth to generate positions for.

        Returns:
            positions (List[List[int]]): A list of positions.
        """
        raise NotImplementedError()

    def change(self, pos: Deque[int], expr: "Expr") -> "Expr":
        """
        指定された場所に基づいて、式の部分を新しい式に書き換える関数。

        ## 引数
        - `pos` (Deque[int]): 自然数列で、書き換える部分を示す。外側の式から内側の部分に向かって引数を指定する
        - `expr` ("Expr"): 新しい式。指定された場所に置き換える新しいサブ式。
        """
        raise NotImplementedError()

    def _change_recursion(self, pos: Deque[int], expr: "Expr") -> "Expr":
        """
        changeメソッドの再帰的な実装

        ## 引数
        - `pos` (Deque[int]): 自然数列で、書き換える部分を示す。外側の式から内側の部分に向かって引数を指定する
        - `expr` ("Expr"): 新しい式。指定された場所に置き換える新しいサブ式。
        """
        raise NotImplementedError()

    def copy(self) -> "Expr":
        """
        属性が全く等しいコピーを作成する
        """
        raise NotImplementedError()


class Z(Expr):
    """
    0に対応するPRFの式
    __init__(): 引数を取らない。引数を取ったらエラー
    arity(): None (意味する関数は任意の数の引数を取るという意味)
    evaluate(): 任意の数の引数で常に0を返す。
    """

    def __init__(self, *args: Any):
        assert len(args) == 0, "The number of args of Z should be 0."

    def _eq_impl(self, other):
        return True  # Z() は常に等しい

    def _hash_impl(self):
        return hash("Z")  # 固定のハッシュ値を返す

    def evaluate(self, *args: int) -> int:
        return 0

    def tree_string(self, indent: int = 0) -> str:
        return " " * indent + "Z"

    def __str__(self) -> str:
        return "Z()"

    def arity(self):
        assert self.is_valid(), "Error: Invalid semantically"
        return None

    def is_valid(self):
        return True

    def positions(self):
        return [deque([])]

    def change(self, pos: Deque[int], expr: "Expr") -> Expr:
        assert pos == Deque([]), "Error: invalid pos arg in Z.change()."
        return expr

    def _change_recursion(self, pos: Deque[int], expr: "Expr") -> Expr:
        assert pos == Deque([]), "Error: invalid pos arg in Z.change()."
        return expr

    def copy(self):
        return Z()


class S(Expr):
    """
    後者関数(+1)に対応するPRFの式
    __init__(): 引数を取るとError
    arity(): 1
    evaluate(): 引数に1を足して返す。1つ以外の引数でError
    """

    def __init__(self, *args):
        assert len(args) == 0, "The number of args of S should be 0."

    def _eq_impl(self, other):
        return True  # S() も常に等しい

    def _hash_impl(self):
        return hash("S")

    def evaluate(self, *args: int) -> int:
        if len(args) != 1:
            raise InputSizeError(
                f"S.evaluate() got invalid input size {len(args)}"
            )

        return args[0] + 1

    def tree_string(self, indent: int = 0) -> str:
        return " " * indent + "S"

    def __str__(self) -> str:
        return "S()"

    def arity(self):
        assert self.is_valid(), "Error: Invalid semantically"
        return 1

    def is_valid(self):
        return True

    def positions(self):
        return [deque([])]

    def change(self, pos: Deque[int], expr: "Expr") -> Expr:
        assert pos == Deque([]), "Error: invalid pos arg in S.change()."
        return expr

    def _change_recursion(self, pos: Deque[int], expr: "Expr") -> Expr:
        assert pos == Deque([]), "Error: invalid pos arg in S.change()."
        return expr

    def copy(self):
        return S()


class P(Expr):
    """
    射影関数に対応するPRFの式
    __init__(n, i): n<iでエラー
    arity: n
    evaluate(): i番目の引数の値を返す。引数の数がnでないときError.
    """

    def __init__(self, n: int, i: int):
        self.n = n
        self.i = i
        assert self.i <= self.n, "Error: P should be self.i <= self.n"

    def _eq_impl(self, other):
        # n と i が同じなら等しい
        return self.n == other.n and self.i == other.i

    def _hash_impl(self):
        return hash((self.n, self.i))

    def evaluate(self, *args: int) -> int:
        if len(args) != self.n:
            raise InputSizeError(
                f"P.evaluate() got invalid input size {len(args)}"
            )
        return args[self.i - 1]

    def tree_string(self, indent: int = 0) -> str:
        return " " * indent + f"P^{self.n}_{self.i}"

    def __str__(self) -> str:
        return f"P({self.n}, {self.i})"

    def arity(self):
        assert self.is_valid(), "Error: Invalid semantically"
        return self.n

    def is_valid(self, *args):
        return True

    def positions(self):
        return [deque([])]

    def change(self, pos: Deque[int], expr: "Expr") -> Expr:
        assert pos == deque([]), "Error: invalid pos arg in P.change()."
        return expr

    def _change_recursion(self, pos: Deque[int], expr: "Expr") -> Expr:
        assert pos == Deque([]), "Error: invalid pos arg in P.change()."
        return expr

    def copy(self):
        return P(self.n, self.i)


class C(Expr):
    """
    関数合成に対応したPRFの式
    __init__(func, args): 引数が2以上でないとError.
    is_valid():
    - argsの長さがfuncのarityと一致しないとFalse。(つまりfunc=Z()もFalse)
    - argsの要素でZ()を除くもの全てのarityが一致しないとFalse
    arity: argsの要素のarity()。validate()を満たさないとError
    evaluate(): 関数合成を行ったfunc(args)を返す。
    """

    def __init__(self, func: Expr, *args: Expr):
        self.func = func
        self.args: tuple[Expr, ...] = args
        assert len(self.args) > 0, "Error: Args of C should be >= 1"

    def _eq_impl(self, other):
        # func と args の全てが同じなら等しい
        return self.func == other.func and self.args == other.args

    def _hash_impl(self):
        return hash((self.func, tuple(self.args)))

    def evaluate(self, *args: int) -> int:
        try:
            results_args: List[int] = [arg.evaluate(*args) for arg in self.args]
            ret = self.func.evaluate(*results_args)
        except InputSizeError as e:
            raise InputSizeError(f"""{e}
                                 {str(self)} got invalid input size {len(args)}.""")
        except OverflowLimitExceededError as e:
            raise OverflowLimitExceededError(f"""{e}
                                 a calculating value exceeds the limit in {str(self)}""")
        except Exception as e:
            raise Exception(f"""{e} 
                            at {str(self)}""")

        if ret > MAX_EVALUATION_LIMIT:
            raise OverflowLimitExceededError(
                f"a calculating value exceeds the limit in {str(self)}"
            )
        return ret

    def tree_string(self, indent: int = 0) -> str:
        buffer = " " * indent + f"C^{1 + len(self.args)}\n"
        buffer = buffer + self.func.tree_string(indent + 2)
        for arg in self.args:
            buffer = buffer + "\n" + arg.tree_string(indent + 2)
        return buffer

    def __str__(self) -> str:
        args_str = ", ".join(str(arg) for arg in self.args)
        return f"C({str(self.func)}, {args_str})"

    def is_valid(self):
        def all_elements_equal(li: List[int]) -> bool:
            if li == []:
                return True
            else:
                for el in li:
                    if li[0] != el:
                        return False
                return True

        if self.func.is_valid() is False:
            return False

        for arg in self.args:
            if arg.is_valid() is False:
                return False

        if self.func.arity() != len(self.args):
            # the number of args in C should be equal to the number of args of C - 1
            return False

        arity_list_not_none = [
            arg.arity() for arg in self.args if arg.arity() is not None
        ]

        if not all_elements_equal(arity_list_not_none):
            # all arity of remaining args after the first arg should be the same
            return False

        return True

    def arity(self):
        assert self.is_valid(), "Error: Invalid semantically"
        return self.args[0].arity()

    def positions(self):
        positions = [deque([])]
        func_positions = self.func.positions()
        for pos in func_positions:
            pos.appendleft(1)
        positions.extend(func_positions)
        for i, arg in enumerate(self.args):
            arg_positions = arg.positions()
            for pos in arg_positions:
                pos.appendleft(i + 2)
            positions.extend(arg_positions)
        return positions

    def change(self, pos: Deque[int], expr: "Expr") -> Expr:
        return self._change_recursion(pos.copy(), expr)

    def _change_recursion(self, pos: Deque[int], expr: "Expr") -> Expr:
        if pos == deque([]):
            return expr

        arg_id = pos.popleft()
        if arg_id == 1:
            copy_args = [arg.copy() for arg in self.args]
            return C(self.func._change_recursion(pos, expr), *copy_args)
        elif arg_id >= 2:
            copy_args = [arg.copy() for arg in self.args]
            copy_args[arg_id - 2] = copy_args[arg_id - 2]._change_recursion(
                pos, expr
            )
            return C(self.func.copy(), *copy_args)
        else:
            raise ValueError(
                "Error: pos arg of C._change_recursion() is invalid. Positive int is needed."
            )

    def copy(self):
        copy_args = (arg.copy() for arg in self.args)
        return C(self.func.copy(), *copy_args)


class R(Expr):
    """
    原始再帰に対応したPRFの式
    __init__(base, step): 引数が2でないとTypeError
    evaluate(): 原始再帰を行った結果を返す。引数の数がおかしかったらError
    is_valid():
    - step.ariy() == NoneでFalse
    - (base, step) == (None, >=2) or (n, n+2) (nは自然数) でないならFalse
    """

    def __init__(self, base: Expr, step: Expr):
        self.base = base
        self.step = step

    def _eq_impl(self, other):
        # base と step が同じなら等しい
        return self.base == other.base and self.step == other.step

    def _hash_impl(self):
        return hash((self.base, self.step))

    def evaluate(self, *args: int) -> int:
        if len(args) == 0:
            InputSizeError(
                "Error: the number of args of R.evaluate() should be >= 1."
            )
        n = args[0]
        post_args = args[1:]
        try:
            if n == 0:
                return self.base.evaluate(*post_args)
            step_back = R(self.base, self.step)
            step_back_ans = step_back.evaluate(n - 1, *post_args)
            ret = self.step.evaluate(n - 1, step_back_ans, *post_args)
        except InputSizeError as e:
            raise InputSizeError(f"""{e}
                                 {str(self)} got invalid input size {len(args)}.""")
        except OverflowLimitExceededError as e:
            raise OverflowLimitExceededError(f"""{e}
                                 a calculating value exceeds the limit in {str(self)}""")
        except Exception as e:
            raise Exception(f"""{e}
                             at {str(self)}""")
        if ret > MAX_EVALUATION_LIMIT:
            raise OverflowLimitExceededError(
                f"a calculating value exceeds the limit in {str(self)}"
            )
        return ret

    def tree_string(self, indent: int = 0) -> str:
        buffer = " " * indent + "R\n"
        buffer += self.base.tree_string(indent + 2) + "\n"
        buffer += self.step.tree_string(indent + 2)
        return buffer

    def __str__(self) -> str:
        buffer = f"R({str(self.base)}, {str(self.step)})"
        return buffer

    def is_valid(self):
        if self.base.is_valid() is False:
            return False
        if self.step.is_valid() is False:
            return False
        # arity is None when expr is const like Z, S(Z), S(S(Z)), S(S(S(Z))), ...
        if self.step.arity() is None:
            return False
        if (self.base.arity() is None and self.step.arity() < 2) or (
            self.base.arity() is not None
            and self.base.arity() + 2 != self.step.arity()
        ):
            # the arity of the args of R operator are wrong.
            return False
        return True

    def arity(self):
        assert self.is_valid(), "Error: Invalid semantically"
        return (
            self.base.arity() + 1
            if self.base.arity() is not None
            else self.step.arity() - 1
        )

    def positions(self) -> List[deque[int]]:
        positions: list[deque[int]] = [deque([])]
        base_positions = self.base.positions()
        for pos in base_positions:
            pos.appendleft(1)
        positions.extend(base_positions)
        step_positions = self.base.positions()
        for pos in step_positions:
            pos.appendleft(2)
        positions.extend(step_positions)
        return positions

    def change(self, pos: Deque[int], expr: "Expr") -> Expr:
        return self._change_recursion(pos.copy(), expr)

    def _change_recursion(self, pos: Deque[int], expr: "Expr") -> Expr:
        if pos == Deque([]):
            return expr

        arg_id = pos.popleft()
        if arg_id == 1:
            return R(self.base._change_recursion(pos, expr), self.step.copy())
        elif arg_id == 2:
            return R(self.base.copy(), self.step._change_recursion(pos, expr))
        else:
            raise ValueError(
                "Error: invaid pos argument (not 1 or 2) at R._change_recursion()"
            )

    def copy(self):
        return R(self.base.copy(), self.step.copy())


def expr_to_str_rec(lst):
    """
    change all Exprs in nested list to str
    """
    if isinstance(lst, list):
        return [expr_to_str_rec(sub) for sub in lst]
    else:
        return str(lst)
