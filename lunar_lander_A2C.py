import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# Define Actor-Critic Network
class ActorCriticNetwork(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ActorCriticNetwork, self).__init__()
        self.common = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(256, action_dim),
            nn.Softmax(dim=-1),
        )
        self.critic = nn.Linear(256, 1)

    def forward(self, state):
        common_features = self.common(state)
        action_probs = self.actor(common_features)
        value = self.critic(common_features)
        return action_probs, value


# A2C Training Loop
def train_a2c(env, num_episodes=1000, gamma=0.99, lr=0.001):
    input_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model = ActorCriticNetwork(input_dim, action_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for episode in range(num_episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        log_probs = []
        values = []
        rewards = []
        total_reward = 0

        done = False
        while not done:
            action_probs, value = model(state)
            action = torch.multinomial(action_probs, 1).item()
            log_prob = torch.log(action_probs.squeeze(0)[action])

            next_state, reward, done, truncated, info = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            total_reward += reward

            state = next_state

        # Compute Returns and Loss
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)

        log_probs = torch.stack(log_probs)
        values = torch.stack(values).squeeze(-1)
        advantage = returns - values.detach()

        actor_loss = -(log_probs * advantage).mean()
        critic_loss = nn.functional.mse_loss(values, returns)
        loss = actor_loss + critic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward:.2f}")

    return model


# Initialize LunarLander environment
if __name__ == "__main__":
    print("Training A2C on LunarLander-v3")
    env = gym.make("LunarLander-v3")
    trained_model = train_a2c(env, num_episodes=500)
    env.close()
