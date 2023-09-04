# Based on the provided code, let's create the complete version.

import os
import numpy as np
import matplotlib.pyplot as plt

def load_data_from_file(filename):
    """
    Load data from the given filename.
    
    Parameters:
    - filename: Path to the data file
    
    Returns:
    - List of dictionaries containing the data
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Extract data from the lines
    data = []
    for i in range(0, len(lines), 4):  # Assuming 4 lines per entry
        entry = {}
        entry['targets_fetched'] = int(lines[i].split(":")[1].strip())
        entry['complete'] = True if lines[i+1].split(":")[1].strip() == "True" else False
        entry['total_time'] = float(lines[i+2].split(":")[1].strip())
        data.append(entry)

    return data

# Load data from the specified path
happo_cu_data_dir = "/home/ubuntu/autodl_one_layer/envs/EnvDrone/classic_control/happo_cu_data.txt"
frontier_dir = "/home/ubuntu/autodl_one_layer/envs/EnvDrone/classic_control/happo_no_cu_collision.txt"
data_happo = load_data_from_file(happo_cu_data_dir)
data_frontier = load_data_from_file(frontier_dir)

# Extract relevant data for plotting for Happo
targets_fetched_happo = [entry['targets_fetched'] for entry in data_happo]
total_time_complete_happo = [entry['total_time'] for entry in data_happo if entry['complete']]

# Extract relevant data for plotting for Frontier
targets_fetched_frontier = [entry['targets_fetched'] for entry in data_frontier]
total_time_complete_frontier = [entry['total_time'] for entry in data_frontier if entry['complete']]

# Compute the success rate for both Happo and Frontier
successful_tasks_happo = sum([1 for entry in data_happo if entry['complete']])
total_tasks_happo = len(data_happo)
success_rate_happo = (successful_tasks_happo / total_tasks_happo) * 100

successful_tasks_frontier = sum([1 for entry in data_frontier if entry['complete']])
total_tasks_frontier = len(data_frontier)
success_rate_frontier = (successful_tasks_frontier / total_tasks_frontier) * 100

# Making the text elements bold
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"

# Plotting the data with separate boxplots for each list
fig, axs = plt.subplots(2, 1, figsize=(12, 12))

# Boxplot for total_time (only successful tasks)
bp1 = axs[0].boxplot(total_time_complete_happo, positions=[1], vert=False, patch_artist=True, labels=['Happo Completed'], medianprops=dict(color="red", linewidth=2.0))
bp2 = axs[0].boxplot(total_time_complete_frontier, positions=[3], vert=False, patch_artist=True, labels=['Frontier Completed'], medianprops=dict(color="red", linewidth=2.0))

# Custom colors for the boxplots
colors1 = ['green']  # Happo: Completed (Green)
colors2 = ['lightgreen']  # Frontier: Completed (Light Green)
for patch, color in zip(bp1['boxes'], colors1):
    patch.set_facecolor(color)
for patch, color in zip(bp2['boxes'], colors2):
    patch.set_facecolor(color)

# Adjusting the title to display the success rates
axs[0].set_title(f"Distribution of Total Time (Happo Success Rate: {success_rate_happo:.2f}%, Frontier Success Rate: {success_rate_frontier:.2f}%)", pad=5)
axs[0].set_xlabel("Total Time")
axs[0].set_yticks([1, 3])
axs[0].set_yticklabels(['Happo Completed', 'Frontier Completed'])
axs[0].grid(True, which="both", ls="--", c='0.7')

# Histogram for targets_fetched
bins = np.arange(1, max(max(targets_fetched_happo), max(targets_fetched_frontier))+2) - 0.5
axs[1].hist(targets_fetched_happo, bins=bins, edgecolor='black', alpha=0.6, color='green', label='Happo')
axs[1].hist(targets_fetched_frontier, bins=bins, edgecolor='black', alpha=0.6, color='lightgreen', label='Frontier')
axs[1].set_title("Distribution of Targets Fetched across All Maps", pad=5)
axs[1].set_xlabel("Number of Targets Fetched")
axs[1].set_ylabel("Frequency")
axs[1].legend(fontsize='large', borderpad=1.5, labelspacing=1.5)
axs[1].grid(True, which="both", ls="--", c='0.7')

plt.tight_layout()
plt.show()
