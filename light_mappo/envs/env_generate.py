import numpy as np
import random
from queue import Queue
import os

def is_accessible(grid_map, start, target):
    w, h = grid_map.shape
    visited = np.zeros_like(grid_map, dtype=bool)
    q = Queue()
    q.put(start)

    while not q.empty():
        y, x = q.get()
        if (y, x) == target:
            return True
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and grid_map[ny, nx] == 0:
                visited[ny, nx] = True
                q.put((ny, nx))

    return False

def generate_map(w, h, min_obstacles, max_obstacles, min_targets, max_targets, allow_closed_space=True):
    grid_map = np.zeros((h, w))
    grid_map[0, :] = 1
    grid_map[-1, :] = 1
    grid_map[:, 0] = 1
    grid_map[:, -1] = 1

    num_obstacles = random.randint(min_obstacles, max_obstacles)

    for _ in range(num_obstacles):
        while True:
            obstacle_x = random.randint(1, w - 2)
            obstacle_y = random.randint(1, h - 2)
            obstacle_w = random.randint(1, w - obstacle_x - 1)
            obstacle_h = random.randint(1, h - obstacle_y - 1)
            tmp_map = grid_map.copy()
            tmp_map[obstacle_y:obstacle_y + obstacle_h, obstacle_x:obstacle_x + obstacle_w] = 2
            if allow_closed_space or is_accessible(tmp_map, (1, 1), (h - 2, w - 2)):
                grid_map = tmp_map
                break

    num_targets = random.randint(min_targets, max_targets)

    for _ in range(num_targets):
        while True:
            target_x = random.randint(1, w - 2)
            target_y = random.randint(1, h - 2)
            if grid_map[target_y, target_x] == 0 and is_accessible(grid_map, (1, 1), (target_y, target_x)):
                grid_map[target_y, target_x] = 3
                break

    return grid_map

def save_map_to_file(grid_map, file_name):
    np.save(file_name, grid_map)


w = 50
h = 50
min_obstacles = 40
max_obstacles = 1000
min_targets = 3
max_targets = 6
allow_closed_space = False

grid_map = generate_map(w, h, min_obstacles, max_obstacles, min_targets, max_targets, allow_closed_space)
print(grid_map)


maps_dir = 'maps'
num_maps = 1000

if not os.path.exists(maps_dir):
    os.makedirs(maps_dir)

for i in range(num_maps):
    grid_map = generate_map(w, h, min_obstacles, max_obstacles, min_targets, max_targets, allow_closed_space)
    file_name = os.path.join(maps_dir, f'map_{i+1}.npy')
    save_map_to_file(grid_map, file_name)
    print(f'Saved map {i+1} to {file_name}')
