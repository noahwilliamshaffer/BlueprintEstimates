# main.py

import gymnasium as gym
from transformers import pipeline

# --- Gym Test ---

def test_gym_env():
    env = gym.make("CartPole-v1")
    observation, info = env.reset()

    for _ in range(5):
        action = env.action_space.sample()  # Take random action
        observation, reward, terminated, truncated, info = env.step(action)
        print(f"Step: Obs={observation}, Reward={reward}, Done={terminated or truncated}")

        if terminated or truncated:
            observation, info = env.reset()

    env.close()

# --- Transformers Test (NLP pipeline) ---

def test_transformer_pipeline():
    summarizer = pipeline("summarization")
    text = """
    The blueprint outlines a residential structure with 2 floors,
    3 bedrooms, and reinforced concrete foundations. The first
    floor includes an open-plan kitchen and living area, while
    the second floor contains all bedrooms and bathrooms.
    """
    summary = summarizer(text, max_length=30, min_length=10, do_sample=False)
    print("\n--- Summary ---")
    print(summary[0]['summary_text'])


if __name__ == "__main__":
    print("\n[+] Testing Gym Environment...")
    test_gym_env()

    print("\n[+] Testing Transformers NLP Pipeline...")
    test_transformer_pipeline()
