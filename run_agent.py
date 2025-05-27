from stable_baselines3 import PPO
from beamng_env import BeamNGEnv
from gymnasium.wrappers import TimeLimit
import time

# Recreate environment
env = BeamNGEnv()
env = TimeLimit(env, max_episode_steps=1000)

# Load the trained model
model = PPO.load("./models/ppo_beamng_20250527_201509.zip")  # Use your actual filename

# Reset the environment
obs, _ = env.reset()

# Run one episode
done = False
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    print(f"Obs: {obs}, Reward: {reward}")

env.close()
