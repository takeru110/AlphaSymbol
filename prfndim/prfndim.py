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
