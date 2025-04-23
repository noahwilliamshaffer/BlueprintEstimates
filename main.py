# main.py

import gymnasium as gym
import numpy as np
from transformers import pipeline
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from gymnasium import spaces
import random

# --- Custom Gym Environment for Commercial Building Cost Estimation ---
class CommercialBuildingCostEnv(gym.Env):
    def __init__(self):
        super(CommercialBuildingCostEnv, self).__init__()

        # Extended observation space with more commercial building parameters
        self.observation_space = spaces.Box(low=0, high=1, shape=(16,), dtype=np.float32)

        # Action: predicted cost (normalized)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        self.current_sample = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_sample = self._generate_building_sample()
        return self.current_sample['features'], {}

    def step(self, action):
        predicted_cost = action[0] * 20_000_000  # scale from 0-1 to $0-$20M
        true_cost = self.current_sample['true_cost']

        reward = -abs(predicted_cost - true_cost) / 1_000_000
        terminated = True
        truncated = False
        return self.current_sample['features'], reward, terminated, truncated, {}

    def _generate_building_sample(self):
        square_footage = random.randint(5000, 200000)
        floors = random.randint(1, 30)
        total_rooms = random.randint(5, 300)
        hvac_type = random.choice([0, 1, 2])  # 0 = basic, 1 = standard, 2 = advanced
        wall_type = random.choice([0, 1, 2])  # 0 = wood, 1 = concrete, 2 = steel
        roof_type = random.choice([0, 1, 2])  # 0 = asphalt, 1 = metal, 2 = composite
        has_elevator = random.choice([0, 1])
        fire_suppression = random.choice([0, 1])
        region = random.choice([0, 1, 2])  # 0 = rural, 1 = suburban, 2 = urban
        building_type = random.choice([0, 1, 2, 3])  # 0 = office, 1 = retail, 2 = mixed-use, 3 = warehouse
        leed_cert = random.choice([0, 1, 2, 3])  # 0 = none, 1 = silver, 2 = gold, 3 = platinum
        interior_finish = random.choice([0, 1, 2])  # 0 = low, 1 = standard, 2 = premium
        structure_type = random.choice([0, 1, 2])  # 0 = steel, 1 = concrete, 2 = hybrid
        occupancy_type = random.choice([0, 1, 2, 3])  # 0 = office, 1 = retail, 2 = school, 3 = hospital

        features = np.array([
            square_footage / 200000,
            floors / 30,
            total_rooms / 300,
            hvac_type / 2,
            wall_type / 2,
            roof_type / 2,
            has_elevator,
            fire_suppression,
            region / 2,
            building_type / 3,
            leed_cert / 3,
            interior_finish / 2,
            structure_type / 2,
            occupancy_type / 3,
            (square_footage / floors) / 10000 if floors else 0,  # floor area ratio
            (total_rooms / square_footage) if square_footage else 0  # room density
        ], dtype=np.float32)

        base_cost = square_footage * 160
        cost_modifiers = (
            floors * 15000 +
            total_rooms * 3500 +
            hvac_type * 25000 +
            wall_type * 20000 +
            roof_type * 15000 +
            has_elevator * 70000 +
            fire_suppression * 30000 +
            region * 150000 +
            building_type * 75000 +
            leed_cert * 40000 +
            interior_finish * 30000 +
            structure_type * 50000 +
            occupancy_type * 60000
        )

        true_cost = base_cost + cost_modifiers

        return {"features": features, "true_cost": true_cost}


# --- NLP Pipeline (Optional) ---
def test_transformer_pipeline():
    summarizer = pipeline("summarization")
    text = """
    The blueprint outlines a modern hospital building with 4 floors,
    120 rooms, advanced HVAC and sprinkler systems, LEED Gold certification,
    hybrid steel-concrete structure, and rooftop solar infrastructure.
    """
    summary = summarizer(text, max_length=30, min_length=10, do_sample=False)
    print("\n--- Summary ---")
    print(summary[0]['summary_text'])


# --- Main Run ---
if __name__ == "__main__":
    print("[+] Checking Custom Environment...")
    env = CommercialBuildingCostEnv()
    check_env(env, warn=True)

    print("[+] Training PPO Agent on CommercialBuildingCostEnv...")
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)

    print("[+] Evaluating Agent...\n")
    obs, _ = env.reset()
    action, _ = model.predict(obs)
    print(f"Predicted normalized cost: {action[0]:.4f}")

    print("[+] Testing Transformers NLP Pipeline...")
    test_transformer_pipeline()
