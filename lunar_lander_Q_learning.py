import gymnasium as gym
import numpy as np

# Initialize environment
env = gym.make("LunarLander-v3", render_mode=None)

# Hyperparameters
learning_rate = 0.1
discount_factor = 0.99
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.01
num_bins = 20  # Number of bins to discretize each state dimension

# Define state bins
state_bins = [
    np.linspace(-1, 1, num_bins),  # x position
    np.linspace(-1, 1, num_bins),  # y position
    np.linspace(-1, 1, num_bins),  # x velocity
    np.linspace(-1, 1, num_bins),  # y velocity
    np.linspace(-np.pi, np.pi, num_bins),  # angle
    np.linspace(-1, 1, num_bins),  # angular velocity
    [0, 1],  # left leg contact
    [0, 1],  # right leg contact
]

# Initialize Q-table with zeros
action_space_size = env.action_space.n
q_table = np.zeros([num_bins] * len(state_bins) + [action_space_size])


# Function to discretize continuous states
def discretize_state(state):
    state_indices = []
    for i, value in enumerate(state):
        state_indices.append(np.digitize(value, state_bins[i]) - 1)
    return tuple(state_indices)


# Q-Learning algorithm to find the first successful landing
def q_learning():
    global epsilon
    episode = 0
    success_found = False

    while not success_found:
        state, info = env.reset()
        state = discretize_state(state)
        done = False
        total_reward = 0

        while not done:
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            # Take action and observe new state and reward
            next_state, reward, done, truncated, info = env.step(action)
            next_state = discretize_state(next_state)
            total_reward += reward

            # Update Q-value
            best_next_action = np.argmax(q_table[next_state])
            q_table[state + (action,)] += learning_rate * (
                reward
                + discount_factor * q_table[next_state + (best_next_action,)]
                - q_table[state + (action,)]
            )

            state = next_state

            if done or truncated:
                break

        episode += 1
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        # Print progress every 100 episodes
        if episode % 100 == 0:
            print(
                f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}"
            )

        # Check if this episode was successful
        if total_reward >= 200:
            success_found = True
            print(f"\nSuccessful Landing found!")
            print(f"Episode: {episode}, Total Reward: {total_reward}")
            break

    env.close()


# Run Q-learning until the first successful landing is found
q_learning()
