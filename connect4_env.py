import gymnasium as gym
from gymnasium import spaces
import numpy as np

class Connect4Env(gym.Env):
    """
    The purpose of the Connect4Env class is to:
        1. Manage the game board.
        2. Follow Gym’s interface (implement reset(), step(action), render(), etc.).
        3. Reward the agent based on the outcome (win, loss, draw).
    """

    def __init__(self):
        super(Connect4Env, self).__init__()
        self.rows = 6
        self.cols = 7
        self.board = np.zeros((self.rows, self.cols), dtype=int)  # Initialize the empty board
        self.current_player = 1  # 1 = Player 1, 2 = Player 2

        # Define the action space: 7 possible columns to drop a piece
        self.action_space = spaces.Discrete(self.cols)

        # Define the observation space: a 6x7 grid with integers (0, 1, 2)
        self.observation_space = spaces.Box(low=0, high=2, shape=(self.rows, self.cols), dtype=int)

    def reset(self, *, seed=None, options=None):
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.current_player = 1
        return self.board.copy(), {}  # Observation and info

    def step(self, action):
        """Execute a move by dropping a piece into the chosen column."""
        # Check if the action is valid (column is not full)
        if self.board[0, action] != 0:
            return self.board.copy(), -10, True, False, {"error": "Invalid move"}  # Invalid move

        # Drop the piece in the chosen column
        for row in range(self.rows - 1, -1, -1):  # Start from the bottom row
            if self.board[row, action] == 0:
                self.board[row, action] = self.current_player
                break

        # Check if the current player has won
        if self.check_winner(self.current_player):
            return self.board.copy(), 1, True, False, {"winner": self.current_player}  # Player wins

        # Check if the game is a draw (board is full)
        if np.all(self.board != 0):
            return self.board.copy(), 0, True, False, {"winner": 0}  # Draw

        # Switch to the other player
        self.current_player = 3 - self.current_player  # Toggle between Player 1 and 2
        return self.board.copy(), 0, False, False, {}  # Game continues

    def render(self):
        """Print the board for visualization."""
        print(self.board[::-1])  # Flip vertically for better visualization

    def check_winner(self, player):
        """Check if the current player has won the game."""
        # Horizontal check
        for row in range(self.rows):
            for col in range(self.cols - 3):
                if np.all(self.board[row, col:col + 4] == player):
                    return True

        # Vertical check
        for row in range(self.rows - 3):
            for col in range(self.cols):
                if np.all(self.board[row:row + 4, col] == player):
                    return True

        # Positive diagonal check (bottom-left to top-right)
        for row in range(self.rows - 3):
            for col in range(self.cols - 3):
                if all(self.board[row + i, col + i] == player for i in range(4)):
                    return True

        # Negative diagonal check (top-left to bottom-right)
        for row in range(3, self.rows):
            for col in range(self.cols - 3):
                if all(self.board[row - i, col + i] == player for i in range(4)):
                    return True

        return False
