from stable_baselines3 import PPO
from connect4_env import Connect4Env

def test_agent():
    # Load the trained model
    model = PPO.load("connect4_ppo")

    # Create the environment
    env = Connect4Env()
    obs = env.reset()
    done = False

    while not done:
        env.render()  # Show the board

        # Let the AI play
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)

        if done:
            env.render()  # Show final board
            if "winner" in info:
                if info["winner"] == 1:
                    print("Player 1 (AI) wins!")
                elif info["winner"] == 2:
                    print("Player 2 (Random or Opponent) wins!")
                else:
                    print("It's a draw!")
            break

test_agent()
