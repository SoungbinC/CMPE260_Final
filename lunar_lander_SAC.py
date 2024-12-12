from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym

# Create the LunarLanderContinuous-v2 environment
env = make_vec_env("LunarLanderContinuous-v3", n_envs=1)

# Initialize the SAC model
model = SAC("MlpPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=100000)

# Save the trained model
model.save("sac_lunar_lander_continuous")

# Evaluate the trained model
mean_reward, std_reward = evaluate_policy(
    model, env, n_eval_episodes=10, deterministic=True
)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Test the trained model with rendering
if __name__ == "__main__":
    test_env = gym.make("LunarLanderContinuous-v3", render_mode="human")
    obs, info = test_env.reset()
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = test_env.step(action)
        if done or truncated:
            obs, info = test_env.reset()

    test_env.close()
