import functools

LRU_CACHE_SIZE = 1000
OVERFLOW = 1000


class InputSizeError(Exception):
    "the number of inputs is invalid for the arity of then function"

    pass


class PrfSyntaxError(Exception):
    "the syntax of the PRF is invalid"

    pass


class SemanticsError(Exception):
    "the semantics of the PRF cannot defined"

    pass


class OverflowError(Exception):
    "the result of the function is overflowed"

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

    def eval(self, *args: int) -> int:
        raise NotImplementedError()

    def __str__(self) -> str:
        raise NotImplementedError()

    def arity(self):
        raise NotImplementedError()

    def is_valid(self):
        raise NotImplementedError()

    def check_overflow(self, value: int):
        if abs(value) > OVERFLOW:
            raise OverflowError(f"Overflowed: {value} at {str(self)}")


class Z(Expr):
    def eval(self, *args: int) -> int:
        return 0

    def __str__(self) -> str:
        return "Z()"

    @property
    def arity(self):
        return None

    @property
    def is_valid(self):
        return True


class S(Expr):
    def __str__(self) -> str:
        return "S()"

    @property
    def arity(self):
        return 1

    @property
    def is_valid(self):
        return True

    def eval(self, *args: int) -> int:
        if len(args) != 1:
            raise InputSizeError(f"S.eval() got invalid input size {len(args)}")

        ret = args[0] + 1
        self.check_overflow(ret)
        return ret


class P(Expr):
    def __init__(self, n: int, i: int):
        if n < i:
            raise PrfSyntaxError("n must be greater than or equal to i")
        self._n = n
        self._i = i

    def __str__(self) -> str:
        return f"P({self._n}, {self._i})"

    @property
    def arity(self):
        return self._n

    @property
    def is_valid(self):
        return True

    def eval(self, *args: int) -> int:
        if len(args) != self._n:
            raise InputSizeError(f"P.eval() got invalid input size {len(args)}")
        ret = args[self._i - 1]
        self.check_overflow(ret)
        return ret


class C(Expr):
    def __init__(self, *args: Expr):
        if len(args) < 2:
            raise PrfSyntaxError("C() requires at least 2 sub Expr")

        self._base: Expr = args[0]
        self._args: tuple[Expr, ...] = args[1:]
        self._is_valid = self._init_is_valid()
        if self.is_valid:
            self._arity = self._init_arity()
        self.eval = functools.lru_cache(maxsize=LRU_CACHE_SIZE)(self._eval)

    def __str__(self) -> str:
        args_str = ", ".join(str(arg) for arg in self._args)
        return f"C({str(self._base)}, {args_str})"

    @property
    def arity(self):
        if not self.is_valid:
            raise SemanticsError(f"{str(self)} is invalid semantically")
        return self._arity

    def _init_arity(self):
        """This function is called only in __init__()."""
        arity_set = set(arg.arity for arg in self._args)
        if any(ar is not None for ar in arity_set):
            return (arity_set - {None}).pop()
        return None

    @property
    def is_valid(self):
        return self._is_valid

    def _init_is_valid(self):
        if not self._base.is_valid:
            return False
        if not all(arg.is_valid for arg in self._args):
            return False
        if self._base.arity not in (None, len(self._args)):
            return False
        var = set(arg.arity for arg in self._args)
        var.discard(None)
        return len(var) in (0, 1)

    def _eval(self, *args: int) -> int:
        if not self.is_valid:
            raise SemanticsError(f"{str(self)} is invalid semantically")
        if self.arity not in (len(args), None):
            raise InputSizeError(
                f"{str(self)} got invalid input size {len(args)}."
            )
        results_args: list[int] = [arg.eval(*args) for arg in self._args]
        ret = self._base.eval(*results_args)
        self.check_overflow(ret)
        return ret


class R(Expr):
    def __init__(self, *args: Expr):
        if len(args) < 2:
            raise PrfSyntaxError("R() requires at least 2 sub Expr")

        if len(args) % 2 == 0:
            raise PrfSyntaxError("the number of sub Exprs of R() must be odd")

        self._dim: int = (len(args) - 1) // 2  # the num of step functions
        self._term = args[0]
        self._steps: tuple[Expr, ...] = args[1 : self._dim + 1]
        self._bases: tuple[Expr, ...] = args[self._dim + 1 :]
        self._is_valid = self._init_is_valid()
        if self.is_valid:
            self._arity = self._init_arity()
        self.eval = functools.lru_cache(maxsize=LRU_CACHE_SIZE)(self._eval)

    def __str__(self) -> str:
        steps_str = ", ".join(str(step) for step in self._steps)
        bases_str = ", ".join(str(base) for base in self._bases)
        return f"R({str(self._term)}, {steps_str}, {bases_str})"

    def _init_arity(self):
        """This function is called only in __init__()."""
        for base in self._bases:
            if base.arity is not None:
                return base.arity + 1
        for step in self._steps:
            if step.arity is not None:
                return step.arity - self._dim
        return None

    @property
    def arity(self):
        if not self.is_valid:
            raise SemanticsError(f"{str(self)} is invalid semantically")
        return self._arity

    @property
    def is_valid(self):
        return self._is_valid

    def _init_is_valid(self):
        """
        term.arity == len(steps) == len(bases) (== self._dim actually)
        and len(steps) == self._dim + base.arity + 1
        """

        if not all(base.is_valid for base in self._bases):
            return False

        if not all(step.is_valid for step in self._steps):
            return False

        if self._term.arity not in (None, self._dim):
            return False

        base_arity_int = set(base.arity for base in self._bases)
        base_arity_int.discard(None)
        base_arity = base_arity_int.pop() if len(base_arity_int) == 1 else None
        step_arity_int = set(step.arity for step in self._steps)
        step_arity_int.discard(None)
        step_arity = step_arity_int.pop() if len(step_arity_int) == 1 else None
        if step_arity is None:
            return True
        if base_arity is None:
            if step_arity >= self._dim + 1:
                return True
            return False
        if step_arity == base_arity + self._dim + 1:
            return True

    def _eval(self, *args: int) -> int:
        if not self.is_valid:
            raise SemanticsError(f"{str(self)} is invalid semantically")
        if self.arity not in (None, len(args)):
            raise InputSizeError(
                f"{str(self)} got invalid input size {len(args)}."
            )
        n = args[0]
        post_args = args[1:]
        if n == 0:
            inter = tuple(base.eval(*post_args) for base in self._bases)
            ret = self._term.eval(*inter)
        else:
            rec_vec = tuple(base.eval(*post_args) for base in self._bases)
            for i in range(0, n):
                rec_vec = tuple(
                    step.eval(i, *rec_vec, *post_args) for step in self._steps
                )
            ret = self._term.eval(*rec_vec)

        self.check_overflow(ret)
        return ret
