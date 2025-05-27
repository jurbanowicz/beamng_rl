from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from beamng_env import BeamNGEnv
from gymnasium.wrappers import TimeLimit
from datetime import datetime
import os

log_dir = "./logs/"
os.makedirs(log_dir, exist_ok=True)

model_dir = "./models/"
os.makedirs(model_dir, exist_ok=True)

tensorboard_log_dir = os.path.join(log_dir, "tensorboard_logs")
os.makedirs(tensorboard_log_dir, exist_ok=True)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

monitor_filename = os.path.join(log_dir, f"monitor_{timestamp}.csv")

env = BeamNGEnv()
env = Monitor(env, filename=monitor_filename)

env = TimeLimit(env, max_episode_steps=1000)
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log_dir)

model_path = os.path.join(model_dir, f"ppo_beamng_{timestamp}")
try:
    model.learn(
        total_timesteps=1_000_000,
        tb_log_name="PPO_BeamNG"
    )
except KeyboardInterrupt:
    print("\n[INFO] Training interrupted. Saving model...")
    model.save(model_path)
    print(f"[INFO] Model saved to {model_path}")
finally:
    env.close()