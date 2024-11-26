import os
import torch
import gymnasium as gym
import numpy as np
import pandas as pd
from lunar_lander_DQN import DQNetwork

import datetime


# Load the saved model
def load_model(path, input_dim, output_dim):
    model = DQNetwork(input_dim, output_dim)  # Initialize the same architecture
    model.load_state_dict(torch.load(path, weights_only=True))  # Add weights_only=True
    model.eval()  # Set the model to evaluation mode
    return model


def evaluate_model_with_video(
    model,
    num_episodes=100,
    video_path="videos",
    video_name="success_landing",
):
    # Ensure the directory exists
    os.makedirs(video_path, exist_ok=True)

    # Setup video recording environment
    video_env = gym.wrappers.RecordVideo(
        gym.make("LunarLander-v3", render_mode="rgb_array"),
        video_path,
        episode_trigger=lambda ep: True,
        name_prefix=video_name,
    )

    total_rewards = []
    successful_landings = 0
    first_success_episode = None
    recorded_success = False

    try:
        for episode in range(num_episodes):
            state, info = video_env.reset()
            done = False
            total_reward = 0

            while not done:
                # Use the model to select actions greedily
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    action = model(state_tensor).argmax().item()

                next_state, reward, done, truncated, info = video_env.step(action)
                state = next_state
                total_reward += reward

            total_rewards.append(total_reward)

            # Check for successful landing between the poles
            final_x = state[0]
            if -0.5 <= final_x <= 0.5 and total_reward >= 200:
                successful_landings += 1
                if first_success_episode is None:
                    first_success_episode = episode + 1

                # Record only the first successful landing video
                if not recorded_success:
                    print(f"Recording first successful landing for {video_name}")
                    recorded_success = True

            if recorded_success:
                break  # Stop recording after the first success

    finally:
        video_env.close()  # Ensure environment is properly closed

    avg_reward = np.mean(total_rewards)
    return avg_reward, successful_landings, first_success_episode


import shutil

# Clear existing videos directory
if os.path.exists("videos"):
    shutil.rmtree("videos")
os.makedirs("videos")

import warnings

# Suppress specific warnings from gymnasium
warnings.filterwarnings("ignore", message="WARN: Overwriting existing videos at")


# Define input and output dimensions for the models
input_dim = 8  # Adjust this based on LunarLander's observation space
output_dim = 4  # Adjust this based on LunarLander's action space

# Load models
dqn_model_path = "dqn_model.pth"  # Path to your saved DQN model
ddqn_model_path = "ddqn_model.pth"  # Path to your saved DDQN model

dqn_model = load_model(dqn_model_path, input_dim, output_dim)
ddqn_model = load_model(ddqn_model_path, input_dim, output_dim)

# Create a timestamped video folder
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
video_path_dqn = f"videos/dqn_{timestamp}"
video_path_ddqn = f"videos/ddqn_{timestamp}"

# Ensure the directories are unique
if os.path.exists(video_path_dqn):
    video_path_dqn += f"_{len(os.listdir('videos'))}"
if os.path.exists(video_path_ddqn):
    video_path_ddqn += f"_{len(os.listdir('videos'))}"

# Evaluate DQN
print("Evaluating DQN...")
avg_reward_dqn, successes_dqn, first_success_dqn = evaluate_model_with_video(
    dqn_model,
    num_episodes=20,
    video_path="videos/dqn",
    video_name="dqn_success_landing",
)

# Evaluate DDQN
print("Evaluating Double DQN...")
avg_reward_ddqn, successes_ddqn, first_success_ddqn = evaluate_model_with_video(
    ddqn_model,
    num_episodes=20,
    video_path="videos/ddqn",
    video_name="ddqn_success_landing",
)
# Log results
results = pd.DataFrame(
    {
        "Algorithm": ["DQN", "Double DQN"],
        "Avg Reward (Last 100 Episodes)": [avg_reward_dqn, avg_reward_ddqn],
        "Successful Landings (Between Poles)": [successes_dqn, successes_ddqn],
        "First Success Episode (Between Poles)": [
            first_success_dqn,
            first_success_ddqn,
        ],
    }
)

print(results)
results.to_csv("evaluation_results.csv", index=False)


from torch.utils.tensorboard import SummaryWriter

# Set up TensorBoard writers for DQN and DDQN
dqn_writer = SummaryWriter(log_dir="runs/dqn")
ddqn_writer = SummaryWriter(log_dir="runs/ddqn")


def evaluate_model_with_tensorboard(
    model, writer, algo_name, num_episodes=100, render_mode="rgb_array"
):
    # Initialize the environment
    env = gym.make("LunarLander-v3", render_mode=render_mode)

    total_rewards = []
    successful_landings = 0
    first_success_episode = None

    for episode in range(num_episodes):
        state, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            # Use the model to select actions
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = model(state_tensor).argmax().item()

            next_state, reward, done, truncated, info = env.step(action)
            state = next_state
            total_reward += reward

        total_rewards.append(total_reward)

        # Check for successful landing between poles
        final_x = state[0]
        if -0.5 <= final_x <= 0.5 and total_reward >= 200:
            successful_landings += 1
            if first_success_episode is None:
                first_success_episode = episode + 1

        # Log reward to TensorBoard
        writer.add_scalar(f"{algo_name}/Reward", total_reward, episode)

    avg_reward = np.mean(total_rewards)
    writer.add_scalar(f"{algo_name}/Avg Reward", avg_reward, 0)
    writer.add_scalar(f"{algo_name}/Successful Landings", successful_landings, 0)
    if first_success_episode:
        writer.add_scalar(
            f"{algo_name}/First Success Episode", first_success_episode, 0
        )

    # Close the environment
    env.close()
    return avg_reward, successful_landings, first_success_episode


# Evaluate DQN
avg_reward_dqn, successes_dqn, first_success_dqn = evaluate_model_with_tensorboard(
    dqn_model, dqn_writer, "DQN", num_episodes=20
)

# Evaluate DDQN
avg_reward_ddqn, successes_ddqn, first_success_ddqn = evaluate_model_with_tensorboard(
    ddqn_model, ddqn_writer, "Double DQN", num_episodes=20
)
