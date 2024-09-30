from typing import Dict, List, Tuple

import gymnasium as gym
from gymnasium import spaces
from strict_prf import C, Expr, P, R, S, Z


class StrictPrfGameEnv(gym.Env):
    """
    Gym environment for the StrictPrfGame.

    The game involves rewriting an initial expression to match input-output test cases.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        max_p_arity: int = 2,
        expr_depth: int = 2,
        max_c_args: int = 2,
        max_steps: int = 100,
        input_sequence: List[int] = None,
        output_sequence: List[int] = None,
    ):
        super(StrictPrfGameEnv, self).__init__()

        self.max_p_arity = max_p_arity
        self.expr_depth = expr_depth
        self.max_c_args = max_c_args
        self.max_steps = max_steps

        self.input_sequence = (
            input_sequence if input_sequence else [0, 1, 2, 3, 4]
        )
        self.output_sequence = (
            output_sequence if output_sequence else [0, 1, 2, 3, 4]
        )

        self.current_expr = Z()
        self.step_count = 0

        # Define action and observation space
        # Actions are discrete, but we'll encode them as tuples (position, token)
        self.action_space = spaces.Discrete(self.calculate_action_space_size())

        # Observations can be represented as the current expression's string representation
        self.observation_space = spaces.Discrete(1)  # Placeholder

        # Mapping from action index to (position, token)
        self.action_index_to_operation = self.generate_action_mapping()

    def reset(self) -> str:
        """Resets the environment to an initial state and returns an initial observation."""
        self.current_expr = Z()
        self.step_count = 0
        return self.get_observation()

    def step(self, action: int) -> Tuple[str, float, bool, bool, Dict]:
        """
        Executes one time step within the environment.

        Args:
            action (int): The action to take.

        Returns:
            observation (str): The current expression's parenthesized string.
            reward (float): The reward obtained after taking the action.
            done (bool): Whether the episode has ended.
            truncated (bool): Whether the episode was truncated.
            info (dict): Additional information.
        """
        self.step_count += 1

        # Perform the action
        position, token = self.action_index_to_operation[action]
        success = self.apply_action(position, token)

        # Check if the current expression matches the test cases
        is_correct = self.check_expression()

        # Calculate reward
        if is_correct:
            reward = 1.0
            done = True
        elif self.step_count >= self.max_steps:
            reward = 0.0
            done = True
        else:
            reward = -0.01  # Small penalty to encourage efficiency
            done = False

        observation = self.get_observation()
        info = {"success": success}

        return observation, reward, done, False, info

    def render(self, mode="human"):
        """Renders the environment."""
        print(
            f"Current Expression:\n{self.current_expr.parenthesized_string()}"
        )

    def get_observation(self) -> str:
        """Returns the current observation."""
        return self.current_expr.parenthesized_string()

    def apply_action(self, position: List[int], token: str) -> bool:
        """Applies the action to the current expression."""
        try:
            new_subexpr = self.token_to_expr(token)
            self.current_expr = self.replace_subexpr(
                self.current_expr, position, new_subexpr
            )
            return True
        except Exception as e:
            print(f"Error applying action: {e}")
            return False

    def token_to_expr(self, token: str) -> Expr:
        """Converts a token string to an Expr object."""
        if token == "z":
            return Z()
        elif token == "s":
            return S()
        elif token.startswith("p_"):
            parts = token.split("_")
            n = int(parts[1])
            i = int(parts[2])
            return P(n, i)
        elif token.startswith("c_"):
            num_args = int(token.split("_")[1])
            args = [Z() for _ in range(num_args)]
            return C(Z(), *args)
        elif token == "r":
            return R(Z(), Z())
        else:
            raise ValueError(f"Unknown token: {token}")

    def replace_subexpr(
        self, expr: Expr, position: List[int], new_subexpr: Expr
    ) -> Expr:
        """Recursively replaces a sub-expression at the given position."""
        if not position:
            return new_subexpr
        else:
            if isinstance(expr, C):
                index = position[0] - 1  # Adjust for 0-based indexing
                if index == 0:
                    func = self.replace_subexpr(
                        expr.func, position[1:], new_subexpr
                    )
                    return C(func, *expr.args)
                else:
                    args = list(expr.args)
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
                raise ValueError(
                    "Cannot replace sub-expression in terminal node"
                )

    def check_expression(self) -> bool:
        """Checks if the current expression produces the correct outputs for the input sequence."""
        try:
            for inp, expected_out in zip(
                self.input_sequence, self.output_sequence
            ):
                result = self.current_expr.evaluate(inp)
                if result != expected_out:
                    return False
            return True
        except Exception:
            return False

    def generate_action_mapping(self) -> Dict[int, Tuple[List[int], str]]:
        """Generates a mapping from action indices to operations (position, token)."""
        actions = []
        positions = self.generate_positions(
            self.current_expr, [], self.expr_depth
        )
        tokens = self.generate_tokens()

        for pos in positions:
            for token in tokens:
                actions.append((pos, token))

        action_mapping = {i: action for i, action in enumerate(actions)}
        return action_mapping

    def calculate_action_space_size(self) -> int:
        """Calculates the size of the action space."""
        num_positions = self.count_positions(self.current_expr, self.expr_depth)
        num_tokens = len(self.generate_tokens())
        return num_positions * num_tokens

    def generate_positions(
        self, expr: Expr, path: List[int], depth: int
    ) -> List[List[int]]:
        """Generates all possible positions up to a certain depth."""
        if depth == 0:
            return [path]
        positions = [path]
        if isinstance(expr, C):
            positions += self.generate_positions(
                expr.func, path + [1], depth - 1
            )
            for i, arg in enumerate(expr.args, start=2):
                positions += self.generate_positions(arg, path + [i], depth - 1)
        elif isinstance(expr, R):
            positions += self.generate_positions(
                expr.base, path + [1], depth - 1
            )
            positions += self.generate_positions(
                expr.step, path + [2], depth - 1
            )
        return positions

    def count_positions(self, expr: Expr, depth: int) -> int:
        """Counts the number of positions up to a certain depth."""
        return len(self.generate_positions(expr, [], depth))

    def generate_tokens(self) -> List[str]:
        """Generates all possible tokens based on the game parameters."""
        tokens = ["z", "s", "r"]
        for n in range(1, self.max_p_arity + 1):
            for i in range(1, n + 1):
                tokens.append(f"p_{n}_{i}")
        for c_args in range(1, self.max_c_args + 1):
            tokens.append(f"c_{c_args}")
        return tokens

    def close(self):
        """Performs any necessary cleanup."""
        pass


if __name__ == "__main__":
    # Example usage
    env = StrictPrfGameEnv()
    observation = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        env.render()
        action = env.action_space.sample()  # Random action for demonstration
        observation, reward, done, truncated, info = env.step(action)
        total_reward += reward

    print(f"Game finished with total reward: {total_reward}")
