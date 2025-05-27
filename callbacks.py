from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter
import numpy as np

class CustomTensorboardCallback(BaseCallback):
    def __init__(self, log_dir, verbose=0):
        super().__init__(verbose)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.episode_num = 0
        self.reset_logs()

    def reset_logs(self):
        self.episode_reward = 0
        self.episode_length = 0
        self.throttle_log = []
        self.clutch_log = []
        self.rpm_log = []
        self.gear_log = []
        self.acceleration_log = []
        self.prev_speed = 0.0

    def _on_step(self) -> bool:
        obs = self.locals['new_obs'][0]  # Obs from env step
        reward = self.locals['rewards'][0]
        done = self.locals['dones'][0]

        speed, rpm, gear, clutch, throttle = obs
        acceleration = speed - self.prev_speed
        self.prev_speed = speed

        self.episode_reward += reward
        self.episode_length += 1
        self.throttle_log.append(throttle)
        self.clutch_log.append(clutch)
        self.rpm_log.append(rpm)
        self.gear_log.append(gear)
        self.acceleration_log.append(acceleration)

        if done:
            self.writer.add_scalar("Reward/Episode_Total", self.episode_reward, self.episode_num)
            self.writer.add_scalar("Episode/Length", self.episode_length, self.episode_num)
            self.writer.add_scalar("Throttle/Average", np.mean(self.throttle_log), self.episode_num)
            self.writer.add_scalar("Clutch/Average", np.mean(self.clutch_log), self.episode_num)
            self.writer.add_scalar("RPM/Average", np.mean(self.rpm_log), self.episode_num)
            self.writer.add_scalar("Gear/Average", np.mean(self.gear_log), self.episode_num)
            self.writer.add_scalar("Acceleration/Average", np.mean(self.acceleration_log), self.episode_num)

            # Optional: histograms
            self.writer.add_histogram("Throttle/Histogram", np.array(self.throttle_log), self.episode_num)
            self.writer.add_histogram("Gear/Histogram", np.array(self.gear_log), self.episode_num)

            self.episode_num += 1
            self.reset_logs()

        return True

    def _on_training_end(self) -> None:
        self.writer.close()
