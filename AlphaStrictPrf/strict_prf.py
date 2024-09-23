from typing import List


class Expr:
    def evaluate(self, *args: int) -> int:
        """自然数関数としてのPRFを評価する"""
        raise NotImplementedError()

    def tree_string(self, indent: int = 0) -> str:
        """木構造をインデント付きで出力する"""
        raise NotImplementedError()

    def parenthesized_string(self) -> str:
        """木構造を括弧で表現して出力する"""
        raise NotImplementedError()

    def complexity(self) -> float:
        """木構造の複雑さを計算する"""
        raise NotImplementedError()

    def validate_semantic(self):
        """自然数関数の入力の次元が正しいかをチェックする
        AssertionError:
            自然数関数に変換しようとしたとき、引数の数が合わないなどによって変換不可能であるとき
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


class Z(Expr):
    def __init__(self, *args: any):
        self.num_args = len(args)

    def evaluate(self, *args: int) -> int:
        return 0

    def tree_string(self, indent: int = 0) -> str:
        return " " * indent + "Z"

    def parenthesized_string(self) -> str:
        return "Z"

    def complexity(self) -> float:
        return 1.0

    def arity(self):
        self.validate_semantic()
        return None

    def validate_semantic(self):
        return self.num_args == 0


class S(Expr):
    def evaluate(self, *args: int) -> int:
        assert len(args) == 1, "The number of args of S should be 1."
        return args[0] + 1

    def tree_string(self, indent: int = 0) -> None:
        return " " * indent + "S"

    def parenthesized_string(self) -> str:
        return "S"

    def complexity(self) -> float:
        return 1.0

    def arity(self):
        self.validate_semantic()
        return 1

    def validate_semantic(self):
        return True


class P(Expr):
    def __init__(self, n: int, i: int):
        self.n = n  # == len(self.args)
        self.i = i
        assert self.i <= self.n, "Error: P should be self.i <= self.n"

    def evaluate(self, *args: int) -> int:
        assert (
            len(args) == self.n
        ), f"Error: the number of args of P.evaluate() should be {self.n + 1} but now {len(args)}"
        return args[self.i - 1]

    def tree_string(self, indent: int = 0) -> None:
        return " " * indent + f"P^{self.n}_{self.i}"

    def parenthesized_string(self) -> str:
        return f"P^{self.n}_{self.i}"

    def complexity(self) -> float:
        return 1.0

    def arity(self):
        self.validate_semantic()
        return self.n

    def validate_semantic(self):
        return True


class C(Expr):
    def __init__(self, func: Expr, *args: Expr):
        self.func = func
        self.args = args

    def evaluate(self, *args: int) -> int:
        results_args: List[int] = [arg.evaluate(*args) for arg in self.args]
        return self.func.evaluate(*results_args)

    def tree_string(self, indent: int = 0) -> None:
        buffer = " " * indent + f"C^{1 + len(self.args)}\n"
        buffer = buffer + self.func.tree_string(indent + 2)
        for arg in self.args:
            buffer = buffer + "\n" + arg.tree_string(indent + 2)
        return buffer

    def parenthesized_string(self) -> str:
        args_str = ", ".join(arg.parenthesized_string() for arg in self.args)
        return f"C^{1 + len(self.args)}({self.func.parenthesized_string()}, {args_str})"

    def complexity(self) -> float:
        return 1.0

    def validate_semantic(self):
        def all_elements_equal(li: List[int]) -> bool:
            if li == []:
                return True
            else:
                for el in li:
                    if li[0] != el:
                        return False
                return True

        self.func.validate_semantic()

        for arg in self.args:
            arg.validate_semantic()

        assert (
            self.func.arity() == len(self.args)
        ), "Error: the number of args in C should be equal to the number of args of C - 1"
        arity_list_not_none = [
            arg.arity() for arg in self.args if arg.arity() is not None
        ]

        assert all_elements_equal(
            arity_list_not_none
        ), "Error: all arity of remaining args after the first arg should be the same"

    def arity(self):
        self.validate_semantic()
        return self.args[0].arity()


class R(Expr):
    def __init__(self, base: Expr, step: Expr):
        self.base = base
        self.step = step

    def evaluate(self, n: int, *args: int) -> int:
        if n == 0:
            return self.base.evaluate(*args)
        else:
            step_back = R(self.base, self.step)
            return self.step.evaluate(
                n - 1, step_back.evaluate(n - 1, *args), *args
            )

    def tree_string(self, indent: int = 0) -> str:
        buffer = " " * indent + "R\n"
        buffer += self.base.tree_string(indent + 2) + "\n"
        buffer += self.step.tree_string(indent + 2)
        return buffer

    def parenthesized_string(self) -> str:
        buffer = f"R({self.base.parenthesized_string()}, {self.step.parenthesized_string()})"
        return buffer

    def complexity(self) -> float:
        return 1.0

    def validate_semantic(self):
        assert (self.base.arity() is not None) & (
            self.base.arity() + 2 == self.step.arity()
        ), "Error: the arity of the args of R operator are wrong."

    def arity(self):
        self.validate_semantic()
        return self.base.arity() + 1
