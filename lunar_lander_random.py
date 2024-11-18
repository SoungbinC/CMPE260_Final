import gymnasium as gym
import numpy as np

# Initialize the Lunar Lander environment
env = gym.make("LunarLander-v3", render_mode=None)

episode = 0
success_found = False

while not success_found:
    state, info = env.reset()
    done = False
    total_reward = 0

    # Run the episode
    while not done:
        # Take a random action
        action = env.action_space.sample()
        next_state, reward, done, truncated, info = env.step(action)
        total_reward += reward

        # Check if the episode is done or truncated
        if done or truncated:
            break

    episode += 1

    # Check if the episode was successful (reward >= 200)
    if total_reward >= 200:
        success_found = True
        print(f"\nSuccessful Landing found!")
        print(f"Total Reward: {total_reward}")
        print(f"Number of Episodes Required: {episode}")
        break

env.close()
