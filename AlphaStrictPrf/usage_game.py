# example_usage.py

from strict_prf_game import StrictPrfGame

# ゲーム環境の初期化
game = StrictPrfGame(
    max_p_arity=2,
    expr_depth=2,
    max_c_args=2,
    max_steps=100,
    input_sequence=[0, 1, 2],
    output_sequence=[0, 1, 2],
)

# ゲームのリセット
observation = game.reset()
done = False
total_reward = 0.0

while not done:
    game.render()
    # 利用可能なアクションを取得
    actions = game.available_actions()
    # ここではランダムなアクションを選択（実際のエージェントではポリシーに基づいて選択）
    action = actions[0]  # 例として最初のアクションを選択
    # アクションを適用
    observation, reward, done, info = game.step(action)
    total_reward += reward

print(f"Game finished with total reward: {total_reward}")
