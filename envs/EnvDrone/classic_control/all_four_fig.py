import numpy as np
import matplotlib.pyplot as plt

# Function to load data from a given file
def load_data_from_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    data = []
    for i in range(0, len(lines), 4):
        entry = {}
        entry['targets_fetched'] = int(lines[i].split(":")[1].strip())
        entry['complete'] = True if lines[i+1].split(":")[1].strip() == "True" else False
        entry['total_time'] = float(lines[i+2].split(":")[1].strip())
        data.append(entry)
    return data

# Load data
happo_cu_data_dir = "/home/ubuntu/autodl_one_layer/envs/EnvDrone/data/happo_cu_data.txt"
frontier_dir = "/home/ubuntu/autodl_one_layer/envs/EnvDrone/data/frontier.txt"
happo_cu_data_dir_dynamic = "/home/ubuntu/autodl_one_layer/envs/EnvDrone/data/happo_cu_data_dynamic.txt"
frontier_dir_dynamic = "/home/ubuntu/autodl_one_layer/envs/EnvDrone/data/frontier_dynamic.txt"
frontier_dir_dynamic_2 = "/home/ubuntu/autodl_one_layer/envs/EnvDrone/data/frontier_dynamic_2.txt"
happo_no_cu_collision = "/home/ubuntu/autodl_one_layer/envs/EnvDrone/data/happo_no_cu_collision.txt"
happo_no_cu_no_collision_200 = "/home/ubuntu/autodl_one_layer/envs/EnvDrone/data/happo_no_cu_no_collision_200.txt"

data_happo_cu = load_data_from_file(happo_cu_data_dir)
data_frontier = load_data_from_file(frontier_dir)
data_happo_no_cu_no_collision = load_data_from_file(happo_no_cu_no_collision_200)
data_happo_no_cu_collision = load_data_from_file(happo_no_cu_collision)


# Extract data for plotting
targets_fetched_happo_cu = [entry['targets_fetched'] for entry in data_happo_cu]
total_time_happo_cu = [entry['total_time'] for entry in data_happo_cu]
targets_fetched_frontier = [entry['targets_fetched'] for entry in data_frontier]
total_time_frontier = [entry['total_time'] for entry in data_frontier]
targets_fetched_happo_no_cu_no_collision = [entry['targets_fetched'] for entry in data_happo_no_cu_no_collision]
total_time_happo_no_cu_no_collision = [entry['total_time'] for entry in data_happo_no_cu_no_collision]
targets_fetched_happo_no_cu_collision = [entry['targets_fetched'] for entry in data_happo_no_cu_collision]
total_time_happo_no_cu_collision = [entry['total_time'] for entry in data_happo_no_cu_collision]

# Calculate success rates
success_rate_happo_cu = (sum([1 for entry in data_happo_cu if entry['complete']]) / len(data_happo_cu)) * 100
success_rate_frontier = (sum([1 for entry in data_frontier if entry['complete']]) / len(data_frontier)) * 100
success_rate_happo_no_cu_no_collision = (sum([1 for entry in data_happo_no_cu_no_collision if entry['complete']]) / len(data_happo_no_cu_no_collision)) * 100
success_rate_happo_no_cu_collision = (sum([1 for entry in data_happo_no_cu_collision if entry['complete']]) / len(data_happo_no_cu_collision)) * 100

# Binning data
bins = np.arange(1, max(
    max(targets_fetched_happo_cu), 
    max(targets_fetched_frontier),
    max(targets_fetched_happo_no_cu_no_collision),
    max(targets_fetched_happo_no_cu_collision)
) + 2) - 0.5

# Plotting the data

# Boxplot data
fig, axs = plt.subplots(1, 2, figsize=(20, 8))

positions = [1, 2, 3, 4]
labels = ['Happo CU', 'Frontier', 'No CU No Collision', 'No CU']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

box_data = [total_time_happo_cu, total_time_frontier, total_time_happo_no_cu_no_collision, total_time_happo_no_cu_collision]
bp = axs[0].boxplot(box_data, positions=positions, vert=False, patch_artist=True, labels=labels, medianprops=dict(color="black", linewidth=2.0))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    
axs[0].set_title(f"Distribution of Total Time", pad=10)
axs[0].set_xlabel("Total Time")
axs[0].grid(True, which="both", ls="--", c='0.7')

# # Adding success rates
# for i, rate in enumerate([success_rate_happo_cu, success_rate_frontier, success_rate_happo_no_cu_no_collision, success_rate_happo_no_cu_collision]):
#     axs[0].text(1.02, positions[i], f"SR: {rate:.2f}%", transform=axs[0].get_yaxis_transform(), ha="left", va="center", color=colors[i])

# Adding success rates
# Adding success rates with further adjustment to place them lower
for i, rate in enumerate([success_rate_happo_cu, success_rate_frontier, success_rate_happo_no_cu_no_collision, success_rate_happo_no_cu_collision]):
    axs[0].text(0, positions[i]-0.2, f"SR: {rate:.2f}%", transform=axs[0].get_yaxis_transform(), ha="right", va="center", color=colors[i])


# Line plot for the second figure
x_values = list(range(1, len(bins)))

# Plotting the datasets as line plots
axs[1].plot(x_values, np.histogram(targets_fetched_happo_cu, bins=bins)[0], color=colors[0], marker='o', label='Happo CU')
axs[1].plot(x_values, np.histogram(targets_fetched_frontier, bins=bins)[0], color=colors[1], marker='o', label='Frontier')
axs[1].plot(x_values, np.histogram(targets_fetched_happo_no_cu_no_collision, bins=bins)[0], color=colors[2], marker='o', label='Happo No CU No Collision')
axs[1].plot(x_values, np.histogram(targets_fetched_happo_no_cu_collision, bins=bins)[0], color=colors[3], marker='o', label='Happo No CU Collision')

# Setting titles, labels, and other plot properties
axs[1].set_title("Number of Targets Fetched per Episode", pad=10)
axs[1].set_xlabel("Number of Targets Fetched")
axs[1].set_ylabel("Map Number")
axs[1].legend(fontsize='large', borderpad=1.5, labelspacing=1.5, loc='upper left')
axs[1].grid(True, which="both", ls="--", c='0.7')
axs[1].set_xticks(x_values)

plt.subplots_adjust(wspace=10)  # Adjust this value as needed
plt.tight_layout()
plt.show()
