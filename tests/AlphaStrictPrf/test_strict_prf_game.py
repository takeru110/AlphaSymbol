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


# メソッドをテストする関数（例外処理のテスト）
def test_get_observation(game_instance):
    assert game_instance.get_observation() == {
        "expression": "Z()",
        "step_count": 0,
    }, "Error: get_observation"


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
        Action(deque([]), R(Z(), Z())),
        Action(deque([]), P(1, 1)),
        Action(deque([]), P(2, 1)),
        Action(deque([]), P(2, 2)),
        Action(deque([]), C(Z(), Z())),
        Action(deque([]), C(Z(), Z(), Z())),
    ]
    processed_expected_actions = [
        Action(tuple(action.position), action.expr)
        for action in expected_actions
    ]
    processed_actions = [
        Action(tuple(action.position), action.expr) for action in actions
    ]
    assert (
        processed_actions == processed_expected_actions
    ), "Error: available_actions"


def test_reset(game_instance):
    state, info = game_instance.reset()
    expected_state = (
        [90, 40, 41]  # ascii of "Z()"
        + [0] * 19997
    )

    expected_info = {"expression": "Z()", "step_count": 0}
    assert state == expected_state, "Error: state of StrictPrfGame.reset()"
    assert info == expected_info, "Error: info of StrictPrfGame.reset()"
