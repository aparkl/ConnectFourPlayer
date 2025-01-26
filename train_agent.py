from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from connect4_env import Connect4Env

# create the Connect 4 environment
env = Connect4Env()

# wrap the environment to support vectorized training
vec_env = make_vec_env(lambda: env, n_envs=1)

model = PPO("MlpPolicy", vec_env, verbose=1)

# Train the agent
model.learn(total_timesteps=999999)

# Save the trained model
model.save("connect4_ppo")
print("Model saved.")