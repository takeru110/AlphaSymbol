import logging
import random
from collections import deque, namedtuple
from typing import Any, Dict, List, Optional

from AlphaStrictPrf.strict_prf import C, Expr, P, R, S, Z

Action = namedtuple("Action", ["position", "expr"])


class ActType:
    def __init__(self, p, c, d):
        self.n = c**d * (3 + int(1 / 2 * p * (p + 1)) + c)
        self.n_post_expr = 3 + int(1 / 2 * p * (p + 1)) + c

    def sample(self):
        return random.randint(0, self.n - 1)


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
        n_obs: int = 1000,
        init_expr: Expr = Z(),
    ):
        self.max_p_arity = max_p_arity
        self.expr_depth = expr_depth
        self.max_c_args = max_c_args
        self.max_steps = max_steps

        self.input_sequence: list[int] = (
            input_sequence if input_sequence else [0, 1, 2]
        )
        self.output_sequence: list[int] = (
            output_sequence if output_sequence else [0, 1, 2]
        )
        self.init_expr = init_expr

        self.current_expr: Expr = init_expr
        self.action_space = ActType(
            self.max_p_arity, self.max_c_args, self.expr_depth
        )

        # Generate possible tokens based on game parameters
        self.tokens = self.available_tokens()
        # like a size of display which will be input of DQN model
        self.n_obs = n_obs

    def reset(self):
        """
        Resets the environment to an initial state and returns an initial observation.
        """
        self.current_expr: Expr = self.init_expr
        self.step_count = 0
        self.done = False

        info = self._get_info()
        state = self._string_to_state(info["expression"])
        return state, info

    def _string_to_state(self, st: str):
        # string to state
        ascii_list = [ord(char) for char in st]
        ascii_list = ascii_list[: self.n_obs]
        ascii_list.extend([0] * (self.n_obs - len(ascii_list)))
        return ascii_list

    def _set_current_expression(self, expr: Expr):
        self.current_expr = expr

    def render(self):
        """
        Renders the environment.

        This method prints the current expression.
        """
        print(f"Step: {self.step_count}")
        print(f"Current Expression:\n{str(self.current_expr)}")

    def current_output(self):
        """
        Evaluates the output of the current expression on the input sequence.
        Returns:
            output (List[int]): The output of the current expression on the input sequence.
                when the expression is invalid, returns [-1, -1, ...]
        """
        try:
            output = [
                self.current_expr.evaluate(i) for i in self.input_sequence
            ]
        except AssertionError:
            output = [-1] * len(self.output_sequence)
        return output

    def _get_info(self) -> Dict[str, Any]:
        """
        Returns the current observation.

        Returns:
            observation (dict): The current observation.
        """
        output = self.current_output()

        info = {
            "expression": str(self.current_expr),
            "step_count": self.step_count,
            "input": self.input_sequence,
            "target": self.output_sequence,
            "output": output,
        }
        return info

    def available_tokens(self) -> List[Expr]:
        """
        Generates all possible expressions (tokens) based on the game parameters.

        Returns:
            tokens (List[Expr]): A list of Expr objects representing valid expressions after substituting.
        """
        tokens = [
            Z(),
            S(),
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

        tokens.append(R(Z(), Z()))
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

    def step_human_readable(
        self, action: Action
    ) -> tuple[str, float, bool, bool, dict[str, Any]]:
        logging.debug("Starting step_human_readable method")
        self.step_count += 1
        length_score = 0.9 ** len(str(self.current_expr))  # in 0 < x < 1
        truncated = self.step_count >= self.max_steps
        pos = action.position
        exp = action.expr

        # position is invalid
        if pos not in self.available_positions():
            logging.debug("Invalid position detected")
            return (
                self.current_expr,
                0 + length_score,
                False,
                truncated,
                self._get_info(),
            )

        # new expression is accepted as next expression
        self.current_expr: Expr = self.current_expr.change(pos, exp)
        length_score = 0.9 ** len(str(self.current_expr))  # in 0 < x < 1

        # Next expression is invalid semantically
        if not self.current_expr.validate_semantic():
            logging.debug("Semantic validation failed")
            return (
                self.current_expr,
                0.1 + length_score,
                False,
                truncated,
                self._get_info(),
            )
        if self.current_expr.arity() != 1:
            logging.debug("Arity is not equal to 1")
            return (
                self.current_expr,
                0.2 + length_score,
                False,
                truncated,
                self._get_info(),
            )

        length_score = 0.9 ** len(str(self.current_expr))

        matching_elements = sum(
            1
            for t, e in zip(self.output_sequence, self.input_sequence)
            if t == self.current_expr.evaluate(e)
        )
        if matching_elements == len(self.input_sequence):
            logging.debug("All output elements matched")
            return (
                self.current_expr,
                1 + length_score,
                True,
                truncated,
                self._get_info(),
            )

        # 0 to 0.3
        correctness_score = 0.3 * matching_elements / len(self.input_sequence)
        logging.debug("Correctness score: {correctness_score}")
        return (
            self.current_expr,
            0.3 + correctness_score + length_score,
            False,
            truncated,
            self._get_info(),
        )

    def int2action(self, num: int) -> Action:
        """Convert int which expresses action into Action

        このメソッドは引数としてint型を受け、Action型(自然数のリストplaceとExpr型変数exprの組)を返す。
        int型をself.action_space.n_post_exprで割ったとき、商がplaceを決め、あまりがexprを決める。

        placeはこの商をmax_c_args + 1進数で表したときの各桁の数値をリストにしたものである。
        exprは、仕様書に書いてある書き換え先変数の値 Z(), S(), P(1, 1), P(2, 1), P(2, 2), P(3, 1), P(3, 2),...,C(Z(), Z()), C(Z(), Z(), Z()), ..., R(Z(), Z())
        をこの順に0, 1, ...,self.action_space.n_post_expr - 1, という値を対応させ、余りに対応する式がexprである。

        Args
            int: 行動を表す自然数
        Returns:
            Action: 行動を表すAction型
        """

        # まず、numをself.action_space.n_post_exprで割る
        quotient, remainder = divmod(num, self.action_space.n_post_expr)

        # placeの計算: quotientを(max_c_args + 1)進数のリストに変換
        place = deque([])
        base = self.max_c_args + 1
        while quotient > 0:
            place.append(quotient % base)
            quotient //= base

        place.reverse()
        # exprの計算: remainderが対応するexprに対応
        expr = self.available_tokens()[remainder]
        return Action(place, expr)

    def generate_state(self) -> List[int]:
        """
        generate state which is used as input to DNN model
        """
        ret = self.input_sequence + self.output_sequence + self.current_output()
        assert (
            len(ret) + len(str(self.current_expr)) <= self.n_obs
        ), "Error: self.current_expr is too long and larger than state window length."
        st = "".join(str(self.current_expr)).ljust(self.n_obs - len(ret))
        ret.extend([ord(x) for x in st])
        return ret

    def step(self, input: int):
        """
        step function for DNN model
        Returns:
        - state: Express current state for DNN
        - reward: Reward for RL
        - terminated: Current state is goal
        - truncated: Terminated do too many attempts.
        - info: other information
        """
        action = self.int2action(input)
        expr, reward, terminated, truncated, info = self.step_human_readable(
            action
        )
        state = self.generate_state()
        info["position"] = action.position
        info["substitute"] = str(action.expr)
        return state, reward, terminated, truncated, info
