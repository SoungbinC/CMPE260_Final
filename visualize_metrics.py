import json
import matplotlib.pyplot as plt
import numpy as np

# Load metrics from JSON file
with open("metrics.json", "r") as f:
    metrics = json.load(f)


# Smoothing function for rewards
def smooth(data, window_size=50):
    smoothed = np.convolve(data, np.ones(window_size) / window_size, mode="valid")
    return smoothed


# Directory to save plots
plot_save_dir = "plots"
os.makedirs(plot_save_dir, exist_ok=True)

# Create and save individual plots for each algorithm
for algo, data in metrics.items():
    rewards = data["rewards"]
    smoothed_rewards = smooth(rewards)  # Smooth rewards for better visualization

    plt.figure(figsize=(10, 6))
    plt.plot(smoothed_rewards, label=f"{algo} (Smoothed)", color="blue")
    plt.title(f"Learning Curve: {algo}")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid()

    # Save the plot as an image file
    plot_file_path = f"{plot_save_dir}/{algo}_learning_curve.png"
    plt.savefig(plot_file_path)
    print(f"Plot for {algo} saved to {plot_file_path}")

    plt.close()  # Close the figure to save memory
