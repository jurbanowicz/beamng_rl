import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from beamng_env import BeamNGEnv
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.callbacks import CheckpointCallback
from callbacks import CustomTensorboardCallback
from datetime import datetime
import os
import torch

parser = argparse.ArgumentParser(description="Train PPO agent in BeamNG environment.")
parser.add_argument('--resume', action='store_true', help="Resume training from the latest checkpoint")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

log_dir = "./logs/"
os.makedirs(log_dir, exist_ok=True)

model_dir = "./models/"
os.makedirs(model_dir, exist_ok=True)

tensorboard_log_dir = os.path.join(log_dir, "tensorboard_logs")
os.makedirs(tensorboard_log_dir, exist_ok=True)

custom_tensorboard_log_dir = os.path.join(tensorboard_log_dir, "custom_tensorboard_logs")
os.makedirs(custom_tensorboard_log_dir, exist_ok=True)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

monitor_filename = os.path.join(log_dir, f"monitor_{timestamp}.csv")

env = BeamNGEnv()
env = Monitor(env, filename=monitor_filename)

env = TimeLimit(env, max_episode_steps=1000)

latest_model = None
if args.resume and os.path.isdir(model_dir):
    checkpoints = [f for f in os.listdir(model_dir) if f.endswith(".zip")]
    if checkpoints:
        latest_model = max(checkpoints, key=lambda f: os.path.getctime(os.path.join(model_dir, f)))
        print(f"Resuming from: {latest_model}")
    else:
        print("No saved model found. Starting new training.")

if latest_model:
    model = PPO.load(os.path.join(model_dir, latest_model), env=env, tensorboard_log=tensorboard_log_dir, device=device)
else:
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log_dir, device=device)

checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path=model_dir,
    name_prefix="ppo_beamng_checkpoint"
)

custom_log_file = os.path.join(custom_tensorboard_log_dir, f"log_{timestamp}")

custom_callback = CustomTensorboardCallback(log_dir=custom_log_file)

model_path = os.path.join(model_dir, f"ppo_beamng_{timestamp}")
try:
    model.learn(
        total_timesteps=1_000_000,
        tb_log_name="PPO_BeamNG",
        callback=[checkpoint_callback, custom_callback]
    )
except KeyboardInterrupt:
    print("\n[INFO] Training interrupted. Saving model...")
    model.save(model_path)
    print(f"[INFO] Model saved to {model_path}")
finally:
    env.close()