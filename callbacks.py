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

    def _on_training_start(self) -> None:
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir)
            if self.verbose:
                print(f"[CustomTensorboardCallback] Logging to {self.log_dir}")

    def _on_step(self) -> bool:
        # Get latest info from the environment
        infos = self.locals.get("infos", [])
        if infos:
            info = infos[-1]  # Last env in vectorized env or single env
            step = self.num_timesteps

            if self.verbose:
                print(f"[CustomLog] Step {step}: {info}")

            for key in ['throttle', 'clutch', 'gear', 'rpm', 'speed', 'acceleration']:
                if key in info:
                    self.writer.add_scalar(f"Custom/{key.capitalize()}", info[key], step)

            self.writer.flush()  # Ensure data is written to disk

        return True

    def _on_training_end(self) -> None:
        if self.writer:
            self.writer.close()
