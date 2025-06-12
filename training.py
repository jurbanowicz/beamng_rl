from beamngpy import BeamNGpy, Scenario, Vehicle, angle_to_quat
from beamngpy.sensors import Damage, Electrics

import gymnasium as gym
from gymnasium import spaces
import numpy as np

LOW_GEAR = 0 # Neutral could be chnaged to -1 to include reverse
HIGH_GEAR = 6 # 6th gear is the highest

SPEED_REWARD = 0.5  # Reward multiplier for speed
ACC_REWARD = 20  # Reward multiplier for acceleration
DECC_REWARD = 1  # Penalty multiplier for deceleration (negative acceleration)
OVERREV_PENALTY = 1 * 10  # Penalty for over-revving the engine
WRONG_GEAR_PENALTY = 10.0  # Penalty for being in the wrong gear
RIGHT_GEAR_BONUS = 5.0  # Bonus for being in the right gear

class Training:
    def __init__(self, bng: BeamNGpy, scenario: Scenario, vehicle: Vehicle):
        self.bng = bng
        self.scenario = scenario
        self.vehicle = vehicle

        self.damage_sensor = Damage()
        self.vehicle.attach_sensor('damage', self.damage_sensor)
        self.electric_sensor = Electrics()
        self.vehicle.attach_sensor('electrics', self.electric_sensor)

        self.prev_speed = 0.0
        self.acceleration = 0.0

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
            "rpm": obs[1],
            "gear": obs[2],
            "clutch": obs[3],
            "throttle": obs[4],
            "acceleration": self.acceleration,
        }

        return obs, reward, done, info

    def _get_obs(self):
        """Get the current observation from the vehicle."""
        self.vehicle.poll_sensors()

        speed = self.electric_sensor['airspeed']
        speed = speed * 3.6

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
        damage = self.damage_sensor.get('damage', 0)
        speed = obs[0]
        rpm = obs[1]
        gear = obs[2]
        clutch_input = obs[3]
        throttle_input = obs[4]
        self.acceleration = speed - self.prev_speed
        self.prev_speed = speed
        positive_acceleration = max(self.acceleration, 0)
        negative_acceleration = min(self.acceleration, 0)

        if damage > 100:
            return -10.0
    
        if rpm < 100:  # If the engine is stalled
            return -10.0

        reward = 0.0

        # 1. Reward forward speed (mildly)
        reward += speed * SPEED_REWARD # Encourage movement

        # 2. Reward acceleration
        reward += positive_acceleration * ACC_REWARD
        reward += negative_acceleration * DECC_REWARD  # Penalize deceleration

        if 2000 <= rpm <= 7000:
            reward += 1.0
        elif rpm > 7000:  # Over-revving
            reward -= ((rpm - 7000) / 1000) * OVERREV_PENALTY

        # 3. Penalize clutch abuse (engaged clutch + high throttle or high RPM)
        # if clutch_input < 0.5 and throttle_input > 0.5 and rpm > 3000:
        #     reward -= 2.0  # Harsh clutch dump

        # 4. Penalize mismatched gear/speed
        expected_gear = self._suggested_gear(speed)
        if abs(gear - expected_gear) > 1:
            reward -= 1 * WRONG_GEAR_PENALTY
        else:
            reward += 1 * RIGHT_GEAR_BONUS  # Bonus for being in the right gear

        return reward

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


    def _suggested_gear(self, speed):
        if speed < 40:
            return 1
        elif speed < 70:
            return 2
        elif speed < 150:
            return 3
        elif speed < 200:
            return 4
        elif speed < 240:
            return 5
        else:
            return 6