import gymnasium as gym
from beamngpy import BeamNGpy, Scenario, Vehicle
from training import Training

class BeamNGEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # Initialize BeamNG
        self.bng = BeamNGpy('192.168.1.20', 25252)
        self.vehicle = Vehicle('main_car', model='sunburst2', part_config="vehicles/sunburst2/trackday_M.pc", licence='PYTON_RL')
        self.scenario = Scenario('smallgrid', 'rl_training')

        # Create your training logic object
        self.trainer = Training(self.bng, self.scenario, self.vehicle)

        self.action_space = self.trainer.action_space
        self.observation_space = self.trainer.observation_space

        self.bng.open()
        self._init_scenario()

    def _init_scenario(self):
        self.scenario.add_vehicle(self.vehicle,
                                  pos=(0, 0, 0.1),
                                  rot_quat=(0, 0, 0, 1))
        self.scenario.make(self.bng)
        self.bng.scenario.load(self.scenario)
        self.bng.scenario.start()
        # self.vehicle.ai_set_mode('manual')

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.trainer.restart_scenario()
        obs = self.trainer._get_obs()
        return obs, {}

    def step(self, action):
        obs, reward, done, info = self.trainer.step(action)
        terminated = done
        truncated = info.get("TimeLimit.truncated", False)  # Optional

        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        pass

    def close(self):
        self.bng.disconnect()
        # self.bng.close()
