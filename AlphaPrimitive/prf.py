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


class Z(Expr):
    def evaluate(self, *args: int) -> int:
        return 0

    def tree_string(self, indent: int = 0) -> str:
        return " " * indent + "Z"

    def parenthesized_string(self) -> str:
        return "Z"

    def complexity(self) -> float:
        return 1.0


class S(Expr):
    def evaluate(self, x: int) -> int:
        return x + 1

    def tree_string(self, indent: int = 0) -> None:
        return " " * indent + "S"

    def parenthesized_string(self) -> str:
        return "S"

    def complexity(self) -> float:
        return 1.0


class P(Expr):
    def __init__(self, i: int):
        self.i = i

    def evaluate(self, *args: int) -> int:
        return args[self.i - 1]

    def tree_string(self, indent: int = 0) -> None:
        return " " * indent + f"P{self.i}"

    def parenthesized_string(self) -> str:
        return f"P{self.i}"

    def complexity(self) -> float:
        return 1.0


class C(Expr):
    def __init__(self, func: Expr, *args: Expr):
        self.func = func
        self.args = args

    def evaluate(self, *args: int) -> int:
        results_args: List[int] = [arg.evaluate(*args) for arg in self.args]
        return self.func.evaluate(*results_args)

    def tree_string(self, indent: int = 0) -> None:
        buffer = " " * indent + f"C{1 + len(self.args)}\n"
        buffer = buffer + self.func.tree_string(indent + 2)
        for arg in self.args:
            buffer = buffer + "\n" + arg.tree_string(indent + 2)
        return buffer

    def parenthesized_string(self) -> str:
        args_str = ", ".join(arg.parenthesized_string() for arg in self.args)
        return f"C{1 + len(self.args)}({self.func.parenthesized_string()}, {args_str})"

    def complexity(self) -> float:
        return 1.0


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
