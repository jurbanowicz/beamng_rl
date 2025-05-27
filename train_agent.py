from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from beamng_env import BeamNGEnv
from gymnasium.wrappers import TimeLimit
from datetime import datetime
import os

env = BeamNGEnv()

log_dir = "./logs"
env = Monitor(env, filename=os.path.join(log_dir, f"monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"))

env = TimeLimit(env, max_episode_steps=1000)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000000)

# Save the model
model.save(f"models/ppo_beamng_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

env.close()