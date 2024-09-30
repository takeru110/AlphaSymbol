from typing import Any, Dict, List, Optional

from AlphaStrictPrf.strict_prf import C, Expr, P, R, S, Z


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
        self.step_count = 0
        self.done = False

        # Generate possible tokens based on game parameters
        self.tokens = self.generate_tokens()

    def reset(self):
        """
        Resets the environment to an initial state and returns an initial observation.
        """
        self.current_expr = Z()
        self.step_count = 0
        self.done = False

        observation = self.get_observation()
        return observation

    def step(self, action: Dict[str, Any]):
        """
        Executes one time step within the environment.

        Args:
            action (dict): The action to take, should have 'place' and 'word' keys.

        Returns:
            observation (dict): The current observation.
            reward (float): The reward obtained after taking the action.
            done (bool): Whether the episode has ended.
            info (dict): Additional information.
        """
        if self.done:
            raise Exception(
                "Cannot call step() on a finished game. Please reset the environment."
            )

        self.step_count += 1

        # Extract action components
        position = action.get("place")
        word_token = action.get("word")

        # Validate action
        if position is None or word_token is None:
            raise ValueError("Action must contain 'place' and 'word' keys.")

        # Apply the action
        success = self.apply_action(position, word_token)

        # Check if the current expression matches the test cases
        is_correct = self.check_expression()

        # Calculate reward
        if is_correct:
            reward = 1.0
            self.done = True
        elif self.step_count >= self.max_steps:
            reward = 0.0
            self.done = True
        else:
            reward = -0.01  # Small penalty to encourage efficiency

        observation = self.get_observation()
        info = {"success": success}

        return observation, reward, self.done, info

    def render(self):
        """
        Renders the environment.

        This method prints the current expression.
        """
        print(f"Step: {self.step_count}")
        print(
            f"Current Expression:\n{self.current_expr.parenthesized_string()}"
        )

    def get_observation(self) -> Dict[str, Any]:
        """
        Returns the current observation.

        Returns:
            observation (dict): The current observation.
        """
        observation = {
            "expression": self.current_expr.parenthesized_string(),
            "step_count": self.step_count,
        }
        return observation

    def apply_action(self, position: List[int], token: str) -> bool:
        """
        Applies the action to the current expression.

        Args:
            position (List[int]): The position to apply the action.
            token (str): The token representing the new sub-expression.

        Returns:
            success (bool): Whether the action was successfully applied.
        """
        try:
            new_subexpr = self.token_to_expr(token)
            self.current_expr = self.replace_subexpr(
                self.current_expr, position, new_subexpr
            )
            return True
        except Exception as e:
            # Invalid action or replacement; action fails
            print(f"Error applying action: {e}")
            return False

    def token_to_expr(self, token: str) -> Expr:
        """
        Converts a token string to an Expr object.

        Args:
            token (str): The token to convert.

        Returns:
            expr (Expr): The corresponding expression object.
        """
        if token == "z":
            return Z()
        elif token == "s":
            return S()
        elif token.startswith("p_"):
            parts = token.split("_")
            n = int(parts[1])
            i = int(parts[2])
            if n > self.max_p_arity or i > n:
                raise ValueError(f"Invalid P(n, i) with n={n}, i={i}")
            return P(n, i)
        elif token.startswith("c_"):
            num_args = int(token.split("_")[1])
            if num_args > self.max_c_args:
                raise ValueError(
                    f"Number of C arguments exceeds max_c_args: {num_args}"
                )
            args = [Z() for _ in range(num_args)]
            return C(Z(), *args)
        elif token == "r":
            return R(Z(), Z())
        else:
            raise ValueError(f"Unknown token: {token}")

    def replace_subexpr(
        self, expr: Expr, position: List[int], new_subexpr: Expr
    ) -> Expr:
        """
        Recursively replaces a sub-expression at the given position.

        Args:
            expr (Expr): The current expression.
            position (List[int]): The position to replace.
            new_subexpr (Expr): The new sub-expression.

        Returns:
            expr (Expr): The updated expression.
        """
        if not position:
            return new_subexpr
        else:
            if isinstance(expr, C):
                index = position[0]
                if index == 0:
                    func = self.replace_subexpr(
                        expr.func, position[1:], new_subexpr
                    )
                    return C(func, *expr.args)
                else:
                    args = list(expr.args)
                    if index - 1 >= len(args):
                        raise ValueError(
                            "Invalid position index in C arguments."
                        )
                    args[index - 1] = self.replace_subexpr(
                        args[index - 1], position[1:], new_subexpr
                    )
                    return C(expr.func, *args)
            elif isinstance(expr, R):
                index = position[0]
                if index == 1:
                    base = self.replace_subexpr(
                        expr.base, position[1:], new_subexpr
                    )
                    return R(base, expr.step)
                elif index == 2:
                    step = self.replace_subexpr(
                        expr.step, position[1:], new_subexpr
                    )
                    return R(expr.base, step)
                else:
                    raise ValueError("Invalid position in R")
            else:
                if not position:
                    return new_subexpr
                else:
                    raise ValueError(
                        "Cannot replace sub-expression in terminal node"
                    )

    def check_expression(self) -> bool:
        """
        Checks if the current expression produces the correct outputs for the input sequence.

        Returns:
            is_correct (bool): Whether the expression matches the test cases.
        """
        try:
            if not self.current_expr.validate_semantic():
                return False
            arity = self.current_expr.arity()
            for inp, expected_out in zip(
                self.input_sequence, self.output_sequence
            ):
                inputs = [inp] * arity if arity else []
                result = self.current_expr.evaluate(*inputs)
                if result != expected_out:
                    return False
            return True
        except Exception:
            return False

    def generate_positions(
        self, expr: Expr, path: List[int], depth: int
    ) -> List[List[int]]:
        """
        Generates all possible positions up to a certain depth.

        Args:
            expr (Expr): The current expression.
            path (List[int]): The path to the current node.
            depth (int): The remaining depth.

        Returns:
            positions (List[List[int]]): A list of positions.
        """
        if depth == 0:
            return [path]
        positions = [path]
        if isinstance(expr, C):
            positions += self.generate_positions(
                expr.func, path + [0], depth - 1
            )
            for i, arg in enumerate(expr.args):
                positions += self.generate_positions(
                    arg, path + [i + 1], depth - 1
                )
        elif isinstance(expr, R):
            positions += self.generate_positions(
                expr.base, path + [1], depth - 1
            )
            positions += self.generate_positions(
                expr.step, path + [2], depth - 1
            )
        return positions

    def generate_tokens(self) -> List[Expr]:
        """
        Generates all possible expressions (tokens) based on the game parameters.

        Returns:
            tokens (List[Expr]): A list of Expr objects representing valid expressions.
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

    def available_actions(self) -> List[Dict[str, Any]]:
        """
        Generates a list of all possible actions from the current state.

        Returns:
            actions (List[Dict[str, Any]]): A list of possible actions.
        """
        positions = self.generate_positions(
            self.current_expr, [], self.expr_depth
        )
        actions = []
        for pos in positions:
            for token in self.tokens:
                actions.append({"place": pos, "word": token})
        return actions

    def is_done(self) -> bool:
        """
        Returns whether the game is finished.

        Returns:
            done (bool): True if the game is finished, False otherwise.
        """
        return self.done
