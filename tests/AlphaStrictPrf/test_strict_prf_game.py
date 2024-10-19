from collections import deque

import pytest

from AlphaStrictPrf.strict_prf import C, P, R, S, Z
from AlphaStrictPrf.strict_prf_game import (
    Action,
    StrictPrfGame,  # テスト対象のクラスやモジュールをインポート
)


# テスト対象のクラスのインスタンスを用意するためのfixture
@pytest.fixture
def game_instance():
    # 必要な初期化パラメータを設定してインスタンスを生成
    return StrictPrfGame(
        max_p_arity=2,
        expr_depth=2,
        max_c_args=2,
        max_steps=100,
        input_sequence=[0, 1, 2],
        output_sequence=[0, 1, 2],
    )


def test_available_tokens(game_instance):
    """
    Test for the generate_tokens method using set comparison.
    """
    # Call the generate_tokens method
    tokens = game_instance.available_tokens()
    # Expected tokens (as a set of unique Expr objects)
    expected_tokens = {
        Z(),  # Z()
        S(),  # S()
        R(Z(), Z()),  # R(Z(), Z())
        P(1, 1),  # P(1, 1)
        P(2, 1),  # P(2, 1)
        P(2, 2),  # P(2, 2)
        C(Z(), Z()),  # C(Z(), Z())
        C(Z(), Z(), Z()),
    }

    # Convert the available tokens list to a set
    available_tokens_set = set(tokens)

    # Compare the sets
    assert (
        available_tokens_set == expected_tokens
    ), f"Expected tokens: {expected_tokens}, but got: {available_tokens_set}"


def test_available_positions(game_instance):
    current_expression = C(P(2, 1), Z(), C(P(1, 1), S()))
    game_instance._set_current_expression(current_expression)
    positions = game_instance.available_positions()
    expected_positions = [
        [],
        [1],
        [2],
        [3],
        [3, 1],
        [3, 2],
    ]

    pos_comp = set(tuple(sublist) for sublist in positions)
    exp_pos_comp = set(tuple(sublist) for sublist in expected_positions)
    assert set(pos_comp) == set(exp_pos_comp), "Error: generate_positions()"


def test_available_actions(game_instance):
    actions = game_instance.available_actions()
    expected_actions = [
        Action(deque([]), Z()),
        Action(deque([]), S()),
        Action(deque([]), P(1, 1)),
        Action(deque([]), P(2, 1)),
        Action(deque([]), P(2, 2)),
        Action(deque([]), C(Z(), Z())),
        Action(deque([]), C(Z(), Z(), Z())),
        Action(deque([]), R(Z(), Z())),
    ]
    processed_expected_actions = set(
        Action(tuple(action.position), action.expr)
        for action in expected_actions
    )
    processed_actions = set(
        Action(tuple(action.position), action.expr) for action in actions
    )
    assert (
        processed_actions == processed_expected_actions
    ), "Error: available_actions"


def test_reset(game_instance):
    state, info = game_instance.reset()
    expected_state = (
        [90, 40, 41]  # ascii of "Z()"
        + [0] * 997
    )

    assert state == expected_state, "Error: state of StrictPrfGame.reset()"
    assert info["step_count"] == 0, "Error: step_count of StrictPrfGame.reset()"


class TestStepHumanReadable:
    def test_invalid_position(self):
        # Test case for when the position is invalid
        game = StrictPrfGame(
            max_p_arity=2,
            expr_depth=2,
            max_c_args=2,
            max_steps=100,
            input_sequence=[1, 2, 3],
            output_sequence=[2, 3, 4],
            n_obs=100,
            init_expr=C(P(1, 1), S()),
        )
        game.reset()
        action = Action(
            deque([1, 2]), Z()
        )  # Position not in available_positions

        result = game.step_human_readable(action)

        assert result[:4] == (
            game.current_expr,  # Should return the current expression unchanged
            0 + 0.9 ** len(str(game.current_expr)),  # Expected score
            False,
            False,
        )

    def test_semantically_invalid_expr(self):
        # Test case for when the new expression is semantically invalid
        game = StrictPrfGame(
            max_p_arity=2,
            expr_depth=2,
            max_c_args=2,
            max_steps=100,
            input_sequence=[1, 2, 3],
            output_sequence=[2, 3, 4],
            n_obs=100,
            init_expr=C(P(1, 1), S()),
        )
        game.reset()
        action = Action(deque([1]), P(2, 1))

        result = game.step_human_readable(action)

        assert game.current_expr == C(
            P(2, 1), S()
        )  # Current expression changed
        assert result[:4] == (
            C(P(2, 1), S()),  # New expression
            0.1 + 0.9 ** len(str(game.current_expr)),  # Expected score
            False,  # Done flag
            False,  # Truncated flag (step count < max_steps)
        )

    def test_non_unary_expr(self):
        # Test case for when the new expression is not unary (arity != 1)
        game = StrictPrfGame(
            max_p_arity=2,
            expr_depth=2,
            max_c_args=2,
            max_steps=100,
            input_sequence=[1, 2, 3],
            output_sequence=[2, 3, 4],
            n_obs=100,
            init_expr=C(P(1, 1), S()),
        )
        game.reset()
        action = Action(deque([2]), P(2, 1))

        result = game.step_human_readable(action)
        assert game.current_expr == C(
            P(1, 1), P(2, 1)
        )  # Current expression changed

        assert result[:4] == (
            game.current_expr,  # Should return the current expression changed
            0.2 + 0.9 ** len(str(game.current_expr)),  # Expected score
            False,  # Done flag
            False,  # Truncated flag (step count < max_steps)
        )

    def test_correct_expression(self):
        # Test case for when the new expression evaluates correctly
        game = StrictPrfGame(
            max_p_arity=2,
            expr_depth=2,
            max_c_args=2,
            max_steps=100,
            input_sequence=[1, 2, 3],
            output_sequence=[2, 3, 4],
            n_obs=100,
            init_expr=C(P(1, 1), S()),
        )
        game.reset()
        action = Action(deque([]), S())

        result = game.step_human_readable(action)

        assert result[:4] == (
            S(),
            1
            + 0.9
            ** len(str(game.current_expr)),  # Full score for correct output
            True,  # Done flag (perfect match)
            False,  # Truncated flag (step count < max_steps)
        )

    def test_partial_correctness(self):
        game = StrictPrfGame(
            max_p_arity=2,
            expr_depth=2,
            max_c_args=2,
            max_steps=100,
            input_sequence=[1, 2, 3],
            output_sequence=[2, 3, 4],
            n_obs=100,
            init_expr=C(P(1, 1), S()),
        )
        game.reset()
        # Test case for partial correctness
        action = Action(deque([]), C(S(), C(S(), Z())))

        result = game.step_human_readable(action)

        matching_elements = 1
        expected_score = 0.3 + (
            0.3 * matching_elements / len(game.input_sequence)
        )
        expected_score += 0.9 ** len(str(game.current_expr))  # Add length score

        assert result[:4] == (
            C(S(), C(S(), Z())),  # New expression
            expected_score,  # Partial correctness score
            False,  # Done flag
            False,  # Truncated flag (step count < max_steps)
        )


def test_int2action():
    # When generating semantically invalid expression
    input = [1, 2, 3]
    output = [2, 3, 4]
    game = StrictPrfGame(
        2, 2, 3, 100, input, output, n_obs=100, init_expr=C(P(1, 1), S())
    )
    assert game.int2action(0) == Action(deque([]), Z())
    assert game.int2action(1) == Action(deque([]), S())
    assert game.int2action(4) == Action(deque([]), P(2, 2))
    assert game.int2action(6) == Action(deque([]), C(Z(), Z(), Z()))

    assert game.int2action(45 + 0) == Action(deque([1, 1]), Z())
    assert game.int2action(45 + 1) == Action(deque([1, 1]), S())

    assert game.int2action(90 + 4) == Action(deque([2, 2]), P(2, 2))
    assert game.int2action(90 + 6) == Action(deque([2, 2]), C(Z(), Z(), Z()))


def test_generate_state():
    input = [1, 2, 3, 4, 5]
    output = [2, 3, 4, 5, 6]
    init_expr = C(P(1, 1), S())
    n_obs = 30
    game = StrictPrfGame(
        2, 2, 3, 100, input, output, n_obs=n_obs, init_expr=init_expr
    )
    current_output = [2, 3, 4, 5, 6]
    expected_state = []
    expected_state.extend(input)
    expected_state.extend(output)
    expected_state.extend(current_output)
    expected_state.extend([ord(x) for x in str(init_expr)])
    expected_state.extend([ord(" ")] * (n_obs - len(expected_state)))
    state = game.generate_state()
    assert expected_state == state, "Error. Expr.generate_state()"


def test_step():
    game = StrictPrfGame(
        max_p_arity=3,
        expr_depth=4,
        max_c_args=2,
        max_steps=10,
        input_sequence=[1, 2, 3, 4, 5, 6],
        output_sequence=[4, 5, 6, 7, 8, 9],
        n_obs=50,
        init_expr=Z(),
    )
    game.reset()
    # Step 1: Take a step with input 10 and check the result
    state, reward, terminated, truncated, info = game.step(10)

    # Assert conditions for the step
    assert len(state) == game.n_obs, "State length is incorrect"
    assert isinstance(reward, float), "Reward should be a float"
    assert isinstance(terminated, bool), "Terminated should be a boolean"
    assert isinstance(truncated, bool), "Truncated should be a boolean"
    assert "expression" in info, "Info should contain 'expression'"
    assert not terminated, "Terminated should be False at this point"
    assert not truncated, "Truncated should be False at this point"

    # Step 2: Take another step and verify state changes accordingly
    state, reward, terminated, truncated, info = game.step(25)

    # Assert conditions after second step
    assert (
        len(state) == game.n_obs
    ), "State length is incorrect after second step"
    assert isinstance(
        reward, float
    ), "Reward should be a float after second step"
    assert not terminated, "Terminated should still be False"
    assert not truncated, "Truncated should still be False"

    # Further steps to check changes in the game
    for step_input in [8, 30, 12, 78, 89]:
        state, reward, terminated, truncated, info = game.step(step_input)
        assert (
            len(state) == game.n_obs
        ), f"State length is incorrect after step {step_input}"
        assert isinstance(
            reward, float
        ), f"Reward should be a float after step {step_input}"
        assert isinstance(
            terminated, bool
        ), f"Terminated should be a boolean after step {step_input}"
        assert isinstance(
            truncated, bool
        ), f"Truncated should be a boolean after step {step_input}"
        assert (
            "expression" in info
        ), f"Info should contain 'expression' after step {step_input}"


def test_step_flow():
    game = StrictPrfGame(
        max_p_arity=3,
        expr_depth=4,
        max_c_args=2,
        max_steps=10,
        input_sequence=[1, 2, 3, 4, 5, 6],
        output_sequence=[4, 5, 6, 7, 8, 9],
        n_obs=50,
        init_expr=Z(),
    )
    game.reset()
    game.action_space.n

    state, _, terminated, truncated, info = game.step(10)  # [] R(Z(), Z())
    assert (
        str(info["expression"]) == "R(Z(), Z())"
    ), "Error: return value state of SrictPrfGame.step()"
    assert (
        terminated == False
    ), "Error: return value terminated of StrictPrfGame.step()"
    assert (
        truncated == False
    ), "Error: return value truncated of StrictPrfGame.step()"

    state, _, terminated, truncated, info = game.step(25)  # [2] P(2, 1)
    assert (
        str(info["expression"]) == "R(Z(), P(2, 1))"
    ), "Error: return value state of SrictPrfGame.step()"
    assert (
        terminated == False
    ), "Error: return value terminated of StrictPrfGame.step()"
    assert (
        truncated == False
    ), "Error: return value truncated of StrictPrfGame.step()"

    state, _, terminated, truncated, info = game.step(8)  # [] C(Z(), Z())
    assert (
        str(info["expression"]) == "C(Z(), Z())"
    ), "Error: return value state of SrictPrfGame.step()"
    assert (
        terminated == False
    ), "Error: return value terminated of StrictPrfGame.step()"
    assert (
        truncated == False
    ), "Error: return value truncated of StrictPrfGame.step()"

    state, _, terminated, truncated, info = game.step(30)  # [2] C(Z(), Z())
    assert (
        str(info["expression"]) == "C(Z(), C(Z(), Z()))"
    ), "Error: return value state of SrictPrfGame.step()"
    assert (
        terminated == False
    ), "Error: return value terminated of StrictPrfGame.step()"
    assert (
        truncated == False
    ), "Error: return value truncated of StrictPrfGame.step()"

    state, _, terminated, truncated, info = game.step(12)  # [1] S()
    assert (
        str(info["expression"]) == "C(S(), C(Z(), Z()))"
    ), "Error: return value state of SrictPrfGame.step()"
    assert (
        terminated == False
    ), "Error: return value terminated of StrictPrfGame.step()"
    assert (
        truncated == False
    ), "Error: return value truncated of StrictPrfGame.step()"

    state, _, terminated, truncated, info = game.step(78)  # [2, 1] S()
    assert (
        str(info["expression"]) == "C(S(), C(S(), Z()))"
    ), "Error: return value state of SrictPrfGame.step()"
    assert (
        terminated == False
    ), "Error: return value terminated of StrictPrfGame.step()"
    assert (
        truncated == False
    ), "Error: return value truncated of StrictPrfGame.step()"

    state, _, terminated, truncated, info = game.step(89)  # [2, 2] S()
    assert (
        str(info["expression"]) == "C(S(), C(S(), S()))"
    ), "Error: return value state of SrictPrfGame.step()"
    assert (
        terminated == True
    ), "Error: return value terminated of StrictPrfGame.step()"
    assert (
        truncated == False
    ), "Error: return value truncated of StrictPrfGame.step()"
