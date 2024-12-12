import gymnasium as gym
import torch
import json
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from torch.utils.tensorboard import SummaryWriter


# Define A2C Network
class ActorCriticNetwork(torch.nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ActorCriticNetwork, self).__init__()
        self.common = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.ReLU(),
        )
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(256, action_dim),
            torch.nn.Tanh(),  # Continuous actions scaled to [-1, 1]
        )
        self.critic = torch.nn.Linear(256, 1)

    def forward(self, state):
        common_features = self.common(state)
        actions = self.actor(common_features)
        value = self.critic(common_features)
        return actions, value


# A2C Training Loop
def train_a2c(env, num_episodes=1000, gamma=0.99, lr=0.001):
    input_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    model = ActorCriticNetwork(input_dim, action_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    rewards = []
    for episode in range(num_episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        log_probs = []
        values = []
        rewards_per_episode = []
        done = False

        while not done:
            actions, value = model(state)
            actions = actions.squeeze(0).detach().numpy()
            next_state, reward, done, truncated, info = env.step(actions)

            rewards_per_episode.append(reward)
            log_probs.append(actions)
            values.append(value)

            state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

        rewards.append(sum(rewards_per_episode))

    return model, rewards


# Unified Training Function
def train_and_evaluate(
    algorithm, env_id="LunarLander-v2", num_episodes=1000, save_model_path="models"
):
    # Handle discrete and continuous environments
    if algorithm in ["DQN", "DDQN"]:
        env_id = "LunarLander-v2"  # DQN and DDQN require discrete action spaces
    else:
        env_id = "LunarLanderContinuous-v2"

    env = gym.make(env_id)
    rewards = []
    model = None  # Initialize model reference

    if algorithm == "A2C":
        model, rewards = train_a2c(env, num_episodes=num_episodes)
        torch.save(model.state_dict(), f"{save_model_path}/{algorithm}_model.pth")
    elif algorithm == "SAC":
        model = SAC("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=num_episodes * 500)  # SAC uses timesteps
        rewards, _ = evaluate_policy(
            model, env, n_eval_episodes=num_episodes, return_episode_rewards=True
        )
        model.save(f"{save_model_path}/{algorithm}_model")
    elif algorithm == "DQN":
        model = DQN("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=num_episodes * 500)
        rewards, _ = evaluate_policy(
            model, env, n_eval_episodes=num_episodes, return_episode_rewards=True
        )
        model.save(f"{save_model_path}/{algorithm}_model")
    elif algorithm == "DDQN":
        model = DQN(
            "MlpPolicy", env, verbose=1, double_q=True
        )  # Enable Double Q-Learning
        model.learn(total_timesteps=num_episodes * 500)
        rewards, _ = evaluate_policy(
            model, env, n_eval_episodes=num_episodes, return_episode_rewards=True
        )
        model.save(f"{save_model_path}/{algorithm}_model")

    env.close()
    return rewards


# Save Metrics
def save_metrics(metrics, filename="metrics.json"):
    with open(filename, "w") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    algorithms = ["A2C", "SAC", "DQN", "DDQN"]
    all_metrics = {}

    for algo in algorithms:
        print(f"Training {algo}...")
        rewards = train_and_evaluate(algo, num_episodes=1000)
        all_metrics[algo] = {"rewards": rewards}

    save_metrics(all_metrics)

    print("Metrics saved to metrics.json")
