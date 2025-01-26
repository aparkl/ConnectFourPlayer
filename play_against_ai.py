from stable_baselines3 import PPO
from connect4_env import Connect4Env

def play_against_ai():
    # Load the trained model
    model = PPO.load("connect4_ppo")  # Replace with your saved model filename

    # Create the environment
    env = Connect4Env()
    obs = env.reset()
    terminated, truncated = False, False

    print("Welcome to Connect 4!")
    print("You are Player 1. Enter a column number (0-6) to make your move.\n")

    while not (terminated or truncated):
        env.render()  # Show the current state of the board

        if env.current_player == 1:  # Human's turn
            try:
                action = int(input("Enter a column (0-6): "))
                if action < 0 or action >= env.cols or env.board[0, action] != 0:
                    print("Invalid move! Try again.")
                    continue
            except ValueError:
                print("Invalid input! Enter a number between 0 and 6.")
                continue
        else:  # AI's turn
            print("AI is thinking...")
            action, _ = model.predict(obs)

        # Apply the action
        obs, reward, terminated, truncated, info = env.step(action)

        # Check if the game is over
        if terminated or truncated:
            env.render()  # Show final board
            if "winner" in info:
                if info["winner"] == 1:
                    print("Congratulations, you win!")
                elif info["winner"] == 2:
                    print("The AI wins! Better luck next time.")
                else:
                    print("It's a draw!")
            break

if __name__ == "__main__":
    play_against_ai()
