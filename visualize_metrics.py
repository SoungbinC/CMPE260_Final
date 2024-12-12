import json
import matplotlib.pyplot as plt

# Load metrics
with open("metrics.json", "r") as f:
    metrics = json.load(f)

# Plot rewards for all algorithms
plt.figure(figsize=(10, 6))
for algo, data in metrics.items():
    plt.plot(data["rewards"], label=algo)

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Algorithm Comparison on LunarLander")
plt.legend()
plt.grid()
plt.show()
