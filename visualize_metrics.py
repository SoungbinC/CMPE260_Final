import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Load metrics from JSON
with open("metrics.json", "r") as f:
    metrics = json.load(f)


# Smoothing function
def smooth(data, window_size=50):
    if len(data) < window_size:  # Handle small datasets
        return data
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")


# Create directory for plots
plot_save_dir = "plots"
os.makedirs(plot_save_dir, exist_ok=True)

# Generate plots
for algo, data in metrics.items():
    rewards = data.get("rewards", [])
    if not rewards:
        print(f"No rewards found for {algo}. Skipping plot.")
        continue

    smoothed_rewards = smooth(rewards)

    plt.figure(figsize=(10, 6))
    plt.plot(smoothed_rewards, label=f"{algo} (Smoothed)", color="blue")
    plt.title(f"Learning Curve: {algo}")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid()

    # Save plot
    plot_file_path = f"{plot_save_dir}/{algo}_learning_curve.png"
    plt.savefig(plot_file_path)
    print(f"Plot for {algo} saved to {plot_file_path}")

    plt.close()
