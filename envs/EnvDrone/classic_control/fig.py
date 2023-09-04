import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
frontier_dir = "/home/ubuntu/autodl_one_layer/envs/EnvDrone/classic_control/frontier.txt"
data = load_data_from_file(frontier_dir)

# Extract relevant data for plotting
targets_fetched = [entry['targets_fetched'] for entry in data]
total_time_complete = [entry['total_time'] for entry in data if entry['complete']]
total_time_failed = [entry['total_time'] for entry in data if not entry['complete']]

# Calculate the success rate
total_tasks = len(data)
successful_tasks = sum([1 for entry in data if entry['complete']])
success_rate = (successful_tasks / total_tasks) * 100

# Plotting the data with separate boxplots for each list
fig, axs = plt.subplots(2, 1, figsize=(12, 12))

# Boxplot for total_time
axs[0].boxplot(total_time_complete, positions=[1], vert=False, patch_artist=False)
axs[0].boxplot(total_time_failed, positions=[2], vert=False, patch_artist=False)
axs[0].set_title(f"Distribution of Total Time (Success Rate: {success_rate:.2f}%)")
axs[0].set_xlabel("Total Time")
axs[0].set_yticks([1, 2])
axs[0].set_yticklabels(['Completed Tasks', 'Failed Tasks'])
axs[0].grid(True, which="both", ls="--", c='0.7')

# Histogram for targets_fetched
axs[1].hist(targets_fetched, bins=np.arange(1, max(targets_fetched)+2)-0.5, edgecolor='black')
axs[1].set_title("Distribution of Targets Fetched across All Maps")
axs[1].set_xlabel("Number of Targets Fetched")
axs[1].set_ylabel("Frequency")
axs[1].grid(True, which="both", ls="--", c='0.7')

# Adjust layout
plt.tight_layout()
plt.show()
