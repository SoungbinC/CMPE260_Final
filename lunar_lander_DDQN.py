import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os

# Hyperparameters
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
learning_rate = 1e-3
batch_size = 64
target_update_frequency = 10
memory_size = 10000
num_episodes = 1000
model_save_path = "dqn_lunar_lander.pth"
success_log_path = "first_success.txt"
video_save_path = "videos"

# Ensure video directory exists
os.makedirs(video_save_path, exist_ok=True)

# Initialize environment
env = gym.make("LunarLander-v3", render_mode=None)


# Neural Network Model
class DQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.network(x)


# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
        )

    def size(self):
        return len(self.buffer)


# Initialize Networks and Replay Buffer
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
policy_net = DQNetwork(input_dim, output_dim)
target_net = DQNetwork(input_dim, output_dim)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
replay_buffer = ReplayBuffer(memory_size)


# Epsilon-greedy policy
def select_action(state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return policy_net(state).argmax().item()


def train_ddqn():
    if replay_buffer.size() < batch_size:
        return

    # Sample a batch of experiences from the replay buffer
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

    # Current Q-values from the policy network
    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()

    # Double DQN: Action selection from the policy network
    with torch.no_grad():
        next_actions = policy_net(next_states).argmax(1)
        max_next_q_values = (
            target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
        )

    # Calculate target Q-values
    target_q_values = rewards + gamma * max_next_q_values * (1 - dones)

    # Compute loss
    loss = nn.MSELoss()(q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# Save the model
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


# Load the model
def load_model(model, path):
    model.load_state_dict(torch.load(path))
    print(f"Model loaded from {path}")


# Log the first success
def log_first_success(episode):
    with open(success_log_path, "w") as file:
        file.write(f"First successful landing at episode: {episode}\n")
    print(f"First successful landing logged at episode: {episode}")


# Record a successful episode to a video file if it lands between the poles
def record_successful_landing(episode):
    video_env = gym.wrappers.RecordVideo(
        gym.make("LunarLander-v3", render_mode="rgb_array"),
        video_save_path,
        episode_trigger=lambda ep: True,
    )

    state, info = video_env.reset()
    done = False
    total_reward = 0

    while not done:
        action = select_action(
            state, epsilon=0
        )  # Use the trained model (epsilon=0 for greedy action)
        next_state, reward, done, truncated, info = video_env.step(action)
        state = next_state
        total_reward += reward

    video_env.close()

    # Check if the lander is between the poles and the total reward is >= 200
    final_x = state[0]  # x-coordinate of the lander
    if -0.5 <= final_x <= 0.5 and total_reward >= 200:
        print(
            f"Recorded successful landing between poles at episode {episode} with reward {total_reward}"
        )
    else:
        print(
            f"Landing at episode {episode} did not meet the criteria (x={final_x}, reward={total_reward})"
        )


def train_agent_with_metrics(num_episodes, train_fn, save_model_path=None):
    global epsilon
    episode_rewards = []
    successful_landings = 0
    first_success_episode = None

    for episode in range(num_episodes):
        state, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = select_action(state, epsilon)
            next_state, reward, done, truncated, info = env.step(action)

            # Reward shaping: Encourage staying near poles
            if -0.5 <= state[0] <= 0.5:
                reward += 5  # Bonus for being between poles
            else:
                reward -= 5  # Penalty for moving away

            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            train_fn()

            if done or truncated:
                final_x = state[0]

                # Check if the landing is successful
                if total_reward >= 200:
                    if -0.5 <= final_x <= 0.5:
                        successful_landings += 1  # Increment success count

                        # Record the first success episode
                        if first_success_episode is None:
                            first_success_episode = episode + 1
                break

        episode_rewards.append(total_reward)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Print progress every 100 episodes
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(
                f"Episode: {episode + 1}, Avg Reward (Last 100): {avg_reward:.2f}, "
                f"Successful Landings: {successful_landings}, Epsilon: {epsilon:.2f}"
            )

        # Update target network periodically
        if episode % target_update_frequency == 0:
            target_net.load_state_dict(policy_net.state_dict())

    if save_model_path:
        save_model(policy_net, save_model_path)

    # Return metrics for comparison
    avg_reward_last_100 = np.mean(episode_rewards[-100:])
    return avg_reward_last_100, successful_landings, first_success_episode


if __name__ == "__main__":

    print("Training DDQN...")
    avg_reward_ddqn, successes_ddqn, first_success_ddqn = train_agent_with_metrics(
        num_episodes=1000,
        train_fn=train_ddqn,  # Pass DDQN training function
        save_model_path="ddqn_model.pth",
    )
