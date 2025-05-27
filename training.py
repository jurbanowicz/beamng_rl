from beamngpy import BeamNGpy, Scenario, Vehicle, angle_to_quat
from beamngpy.sensors import Damage, Electrics

import gymnasium as gym
from gymnasium import spaces
import numpy as np

LOW_GEAR = 0 # Neutral could be chnaged to -1 to include reverse
HIGH_GEAR = 6 # 6th gear is the highest

class Training:
    def __init__(self, bng: BeamNGpy, scenario: Scenario, vehicle: Vehicle):
        self.bng = bng
        self.scenario = scenario
        self.vehicle = vehicle

        self.damage_sensor = Damage()
        self.vehicle.attach_sensor('damage', self.damage_sensor)
        self.electric_sensor = Electrics()
        self.vehicle.attach_sensor('electrics', self.electric_sensor)

        # Action space: [throttle, clutch, gear]
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0]),     # Throttle, clutch, gear (float, will round)
            high=np.array([1.0, 1.0, 6.0]),
            dtype=np.float32
        )


        # obs vector: [speed, rpm, gear, clutch_input, throttle_input]
        self.observation_space = spaces.Box(low=np.array([0, 0, LOW_GEAR, 0, 0]),
                                    high=np.array([300, 10000, HIGH_GEAR, 1, 1]),
                                    dtype=np.float32)

    # def start(self):
    #     """Start the lerning process in current scenario."""

    #     self.vehicle.control(steering=0.0, throttle=0.0, brake=0.0, gear=2)
    #     obs = self._get_obs()
    #     done = False

    #     # self.bng.control.pause()

    #     while not done:
    #         action = self.action_space.sample()  # Replace with agent later
    #         obs, reward, done, _ = self.step(action)
    #         # print(f"Obs: {obs}, Reward: {reward}")

    def step(self, action):
        throttle, clutch, gear = action
        gear = int(round(gear))  # Discretize gear

        # Optional: clamp gear to allowed values
        gear = np.clip(gear, LOW_GEAR, HIGH_GEAR)

        self.vehicle.control(
            steering=0.0,  # always straight
            throttle=float(throttle),
            clutch=float(clutch),
            gear=int(gear),
            brake=0.0
        )

        self.bng.step(1)
        self.vehicle.poll_sensors()

        obs = self._get_obs()
        reward = self._compute_reward(obs)
        done = self._check_done(obs)

        info = {
            "damage": self.damage_sensor.data['damage'],
            "speed": obs[0],
            "position": obs[1:4].tolist(),
        }

        return obs, reward, done, info

    def _get_obs(self):
        """Get the current observation from the vehicle."""
        self.vehicle.poll_sensors()

        speed = self.electric_sensor['airspeed']
        # print(f"Electrics data: {self.electric_sensor}")
        clutch_input = self.electric_sensor.get('clutch_input', 0.0)  # Default to 0 if clutch not available
        throttle_input = self.electric_sensor.get('throttle_input', 0.0)  # Default to 0 if throttle not available
        rpm = self.electric_sensor.get('rpm', 0)  # Default to 0 if rpm not available
        gear = self.electric_sensor.get('gear', 0)  # Default to 0 if gear not available
        # state = self.vehicle.state
        # pos_x, pos_y, pos_z = state['pos']
        # brake = self.electric_sensor.get('brake', 0.0)  # Default to 0 if brake not available

        return np.array([speed, rpm, gear, clutch_input, throttle_input], dtype=np.float32)


    def _compute_reward(self, obs):
        damage = self.damage_sensor.data['damage']
        rpm = self.electric_sensor.get('rpm', 0)

        if damage > 100:
            return -100.0 
    
        if rpm < 100:  # If the engine is stalled
            return -100.0
        
        return obs[0]  # Reward = forward speed

    def _check_done(self, obs):
        damage = self.damage_sensor.data['damage']
        rpm = self.electric_sensor.get('rpm', 0)

        if damage > 100:
            print(f"Crash detected! Damage: {damage}")
            return True

        if rpm < 100:  # If the engine is stalled
            print(f"Engine stalled! RPM: {rpm}")
            return True
        return False
        
    def restart_scenario(self):
        """Restart the scenario."""
        self.bng.scenario.restart()
        print("Scenario restarted.")