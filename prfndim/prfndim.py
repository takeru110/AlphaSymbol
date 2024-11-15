class InputSizeError(Exception):
    "the number of inputs is invalid for the arity of then function"

    pass


class PrfSyntaxError(Exception):
    "the syntax of the PRF is invalid"

    pass


class Expr:
    _instances: dict[int, int] = {}

    def __new__(cls, *args):
        key = hash((cls, *args))
        if key in cls._instances:
            return cls._instances[key]
        instance = super().__new__(cls)
        cls._instances[key] = instance
        return instance

    def eval(self, *args: int):
        raise NotImplementedError()

    def __str__(self) -> str:
        raise NotImplementedError()


class Z(Expr):
    def eval(self, *args: int) -> int:
        return 0

    def __str__(self) -> str:
        return "Z()"

    @property
    def arity(self):
        return None


class S(Expr):
    def __str__(self) -> str:
        return "S()"

    @property
    def arity(self):
        return 1


class P(Expr):
    def __init__(self, n: int, i: int):
        if n < i:
            raise PrfSyntaxError("n must be greater than or equal to i")
        self._n = n
        self._i = i

    def __str__(self) -> str:
        return f"P({self._n}, {self._i})"


class C(Expr):
    def __init__(self, *args: Expr):
        if len(args) < 2:
            raise PrfSyntaxError("C() requires at least 2 sub Expr")

        self._base: Expr = args[0]
        self._args: tuple[Expr, ...] = args[1:]

    def __str__(self) -> str:
        args_str = ", ".join(str(arg) for arg in self._args)
        return f"C({str(self._base)}, {args_str})"


class R(Expr):
    def __init__(self, *args: Expr):
        if len(args) < 2:
            raise PrfSyntaxError("R() requires at least 2 sub Expr")

        if len(args) % 2 == 0:
            raise PrfSyntaxError("the number of sub Exprs of R() must be odd")

        self._dim: int = (len(args) - 1) // 2
        self._base: Expr = args[0]
        self._args: tuple[Expr, ...] = args[1:]

    def __str__(self) -> str:
        args_str = ", ".join(str(arg) for arg in self._args)
        return f"R({str(self._base)}, {args_str})"
