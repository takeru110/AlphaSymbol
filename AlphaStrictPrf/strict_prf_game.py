from collections import deque, namedtuple
from typing import Any, Dict, List, Optional

from AlphaStrictPrf.strict_prf import C, Expr, P, R, S, Z

Action = namedtuple("Action", ["position", "expr"])


class StrictPrfGame:
    """
    StrictPrfGame environment without using gymnasium.

    The game involves rewriting an initial expression to match input-output test cases.
    """

    def __init__(
        self,
        max_p_arity: int = 2,
        expr_depth: int = 2,
        max_c_args: int = 2,
        max_steps: int = 100,
        input_sequence: Optional[List[int]] = None,
        output_sequence: Optional[List[int]] = None,
    ):
        self.max_p_arity = max_p_arity
        self.expr_depth = expr_depth
        self.max_c_args = max_c_args
        self.max_steps = max_steps

        self.input_sequence = input_sequence if input_sequence else [0, 1, 2]
        self.output_sequence = output_sequence if output_sequence else [0, 1, 2]

        self.current_expr = Z()

        # Generate possible tokens based on game parameters
        self.tokens = self.available_tokens()

    def reset(self):
        """
        Resets the environment to an initial state and returns an initial observation.
        """
        self.current_expr = Z()
        self.step_count = 0
        self.done = False

        observation = self.get_observation()
        return observation

    def _set_current_expression(self, expr: Expr):
        self.current_expr = expr

    def render(self):
        """
        Renders the environment.

        This method prints the current expression.
        """
        print(f"Step: {self.step_count}")
        print(f"Current Expression:\n{str(self.current_expr)}")

    def get_observation(self) -> Dict[str, Any]:
        """
        Returns the current observation.

        Returns:
            observation (dict): The current observation.
        """
        observation = {
            "expression": str(self.current_expr),
            "step_count": self.step_count,
        }
        return observation

    def available_tokens(self) -> List[Expr]:
        """
        Generates all possible expressions (tokens) based on the game parameters.

        Returns:
            tokens (List[Expr]): A list of Expr objects representing valid expressions after substituting.
        """
        tokens = [
            Z(),
            S(),
            R(Z(), Z()),
        ]  # Z(), S(), R(Z(), Z()) をデフォルトで追加

        # P(n, i) の生成
        for n in range(1, self.max_p_arity + 1):
            for i in range(1, n + 1):
                tokens.append(P(n, i))  # P(n, i) は Expr 型のインスタンス

        # C(Expr1, Expr2, ..., Exprn) の生成
        for c_args in range(1, self.max_c_args + 1):
            args = [
                Z() for _ in range(c_args)
            ]  # C関数の引数としてすべてZ()を初期値にする
            tokens.append(
                C(Z(), *args)
            )  # C(Z(), Z(), ...) のように Expr 型を生成

        return tokens

    def available_positions(self) -> List[deque[int]]:
        return self.current_expr.positions()

    def available_actions(self) -> List[Action]:
        """
        Generates a list of all possible actions from the current state.

        Returns:
            actions (List[Action]): A list of possible actions.
        """
        positions = self.available_positions()
        tokens = self.available_tokens()
        return [Action(pos, token) for pos in positions for token in tokens]
