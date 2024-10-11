import logging
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


def test_step_human_readable():
    # When generating semantically invalid expression
    game1 = StrictPrfGame(
        max_p_arity=2,
        expr_depth=2,
        max_c_args=2,
        max_steps=100,
        input_sequence=[1, 2, 3],
        output_sequence=[2, 3, 4],
        n_obs=100,
        init_expr=C(P(1, 1), S()),
    )
    game1.reset()
    action1 = Action(deque([1]), Z())
    ret = game1.step_human_readable(action1)
    ret_str = (str(ret[0]), *ret[1:])
    match ret_str:
        case ("C(Z(), S())", _, False, False, _):
            pass
        case _:
            pytest.fail("Error: StrictPrfGame.step_human_readable()")

    # When generating arity != 1
    game2 = StrictPrfGame(
        max_p_arity=2,
        expr_depth=2,
        max_c_args=2,
        max_steps=100,
        input_sequence=[1, 2, 3],
        output_sequence=[1, 2, 3],
        n_obs=100,
        init_expr=Z(),
    )
    game2.reset()
    action1 = Action(deque([]), P(2, 1))

    ret = game2.step_human_readable(action1)
    ret_str = (str(ret[0]), *ret[1:])
    match ret_str:
        case ("P(2, 1)", _, False, False, _):
            pass
        case _:
            pytest.fail("Error: StrictPrfGame.step_human_readable()")

    # When agent steped too much
    game3 = StrictPrfGame(
        max_p_arity=2,
        expr_depth=2,
        max_c_args=2,
        max_steps=3,
        input_sequence=[1, 2, 3],
        output_sequence=[4, 5, 6],
        n_obs=10,
    )
    game3.reset()
    action1 = Action(deque([]), P(2, 1))
    action2 = Action(deque([]), Z())
    action3 = Action(deque([]), P(1, 1))
    game3.step_human_readable(action1)
    game3.step_human_readable(action2)
    ret = game3.step_human_readable(action3)
    ret_str = (str(ret[0]), *ret[1:])

    match ret_str:
        case ("P(1, 1)", _, False, True, _):
            pass
        case _:
            pytest.fail("Error: StrictPrfGame.step_human_readable()")

    # When generate correct answer
    game4 = StrictPrfGame(
        max_p_arity=2,
        expr_depth=2,
        max_c_args=2,
        max_steps=100,
        input_sequence=[1, 2, 3],
        output_sequence=[1, 2, 3],
        n_obs=100,
    )
    game4.reset()
    action1 = Action(deque([]), P(1, 1))

    ret = game4.step_human_readable(action1)
    ret_str = (str(ret[0]), *ret[1:])
    match ret_str:
        case ("P(1, 1)", _, True, False, _):
            pass
        case _:
            pytest.fail("Error: StrictPrfGame.step_human_readable()")


def test_int2action():
    # When generating semantically invalid expression
    input = [1, 2, 3]
    output = [2, 3, 4]
    game = StrictPrfGame(
        2, 2, 3, 100, input, output, n_obs=100, init_expr=C(P(1, 1), S())
    )
    assert game.int2action(0) == Action([], Z())
    assert game.int2action(1) == Action([], S())
    assert game.int2action(4) == Action([], P(2, 2))
    assert game.int2action(6) == Action([], C(Z(), Z(), Z()))

    assert game.int2action(45 + 0) == Action([1, 1], Z())
    assert game.int2action(45 + 1) == Action([1, 1], S())

    assert game.int2action(90 + 4) == Action([2, 2], P(2, 2))
    assert game.int2action(90 + 6) == Action([2, 2], C(Z(), Z(), Z()))


def test_generate_state():
    input = [1, 2, 3, 4, 5]
    output = [2, 3, 4, 5, 6]
    init_expr = C(P(1, 1), S())
    n_obs = 30
    game = StrictPrfGame(
        2, 2, 3, 100, input, output, n_obs=n_obs, init_expr=init_expr
    )
    expected_state = []
    expected_state.extend(input)
    expected_state.extend(output)
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
    game.action_space.n
    state, _, _, _, info = game.step(10)
    logging.debug(info["expression"])
    state, _, _, _, info = game.step(25)
    logging.debug(info["expression"])
    state, _, _, _, info = game.step(8)
    logging.debug(info["expression"])
    state, _, _, _, info = game.step(30)
    logging.debug(info["expression"])
    state, _, _, _, info = game.step(12)
    logging.debug(info["expression"])
    state, _, _, _, info = game.step(78)
    logging.debug(info["expression"])
    state, _, _, _, info = game.step(89)
    logging.debug(info["expression"])
