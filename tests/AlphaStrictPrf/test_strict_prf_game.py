# test_my_class.py
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


def test_generate_tokens(game_instance):
    """
    Test for the generate_tokens method using set comparison.
    """
    # Call the generate_tokens method
    tokens = game_instance.generate_tokens()
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

    # Convert the generated tokens list to a set
    generated_tokens_set = set(tokens)

    # Compare the sets
    assert (
        generated_tokens_set == expected_tokens
    ), f"Expected tokens: {expected_tokens}, but got: {generated_tokens_set}"


def test_generate_positions(game_instance):
    current_expression = C(P(2, 1), Z(), C(P(1, 1), S()))
    game_instance._set_current_expression(current_expression)
    positions = game_instance.generate_positions()
    expected_positions = [
        [1],
        [1, 1],
        [1, 2],
        [1, 3],
        [1, 3, 1],
        [1, 3, 2],
    ]

    pos_comp = set(tuple(sublist) for sublist in positions)
    exp_pos_comp = set(tuple(sublist) for sublist in expected_positions)
    assert set(pos_comp) == set(exp_pos_comp), "Error: generate_positions()"


def test_available_actions(game_instance):
    actions = game_instance.available_actions()
    import pdb

    pdb.set_trace()
    set_actions = set(actions)
    expected_actions = {
        Action(1, Z()),
        Action(1, S()),
        Action(1, R(Z(), Z())),
        Action(1, P(1, 1)),
        Action(1, P(2, 1)),
        Action(1, P(2, 2)),
        Action(1, C(Z(), Z())),
        Action(1, C(Z(), Z(), Z())),
    }

    assert set_actions == expected_actions, "Error: available_actions"
