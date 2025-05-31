from stable_baselines3.common.callbacks import BaseCallback
import os
from torch.utils.tensorboard import SummaryWriter

class CustomTensorboardCallback(BaseCallback):
    """
    Custom callback for logging additional environment metrics to TensorBoard.
    """

    def __init__(self, log_dir: str, verbose=0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.writer = None
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_reward = 0.0
        self.current_length = 0

    def _on_training_start(self) -> None:
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir)
            if self.verbose:
                print(f"[CustomTensorboardCallback] Logging to {self.log_dir}")

    def _on_step(self) -> bool:
        # Get latest info from the environment
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        if infos:
            info = infos[-1]  # Last env in vectorized env or single env
            step = self.num_timesteps

            if self.verbose:
                print(f"[CustomLog] Step {step}: {info}")

            for key in ['throttle', 'clutch', 'gear', 'rpm', 'speed', 'acceleration']:
                if key in info:
                    value = float(info[key])
                    self.writer.add_scalar(f"Custom/{key.capitalize()}", value, step)

            # Accumulate reward and episode length
            self.current_reward += self.locals['rewards'][-1]
            self.current_length += 1

            # If the episode ended, log episode metrics
            if dones and dones[-1]:
                self.episode_rewards.append(self.current_reward)
                self.episode_lengths.append(self.current_length)

                self.writer.add_scalar("Custom/EpisodeReward", self.current_reward, step)
                self.writer.add_scalar("Custom/EpisodeLength", self.current_length, step)


            self.writer.flush()  # Ensure data is written to disk

        return True

    def _on_training_end(self) -> None:
        if self.writer:
            self.writer.close()
