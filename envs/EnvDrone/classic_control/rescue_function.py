from queue import PriorityQueue
import sys
import numpy as np
sys.path.append(r"d:/code/TRPO-in-MARL/")

import scripts.train.rescue as rescue

class rescue_action():
    def __init__(self, actions, id):
        self.actions = actions
        self.id = id

def find_nearest_frontier(map, start):
    rows, cols = map.shape
    queue = PriorityQueue()
    start = (start[0], start[1])
    queue.put((0, start))
    visited = set()
    visited.add(start)
    last_final_point = None

    while not queue.empty():
        distance, (r, c) = queue.get()
        if map[r, c] == 0:
            return last_final_point, r, c

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc

            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue
            if (nr, nc) in visited:
                continue

            if map[nr, nc] <= 1:
                if map[nr, nc] == 0:  # We are about to visit an unknown cell
                    last_final_point = (r, c)
                queue.put((distance + 1, (nr, nc)))
                visited.add((nr, nc))
    return None


def generate_path(env, id, free_zone, obstacle_map, pos = None):
    start_pos = env.drone_list[id].pos
    if pos is not None:
        start_pos = pos
    # 开始规划路径，第一步找到这个agent的地图里所有的边界点，为此，我们需要一份同时包含障碍物和已探索区域的地图
    # 此时，map中 value>=2 的点就是障碍物，0<value<=1的点就是已经搜索的区域
    map = env.drone_list[id].whole_map[1] + 2 * env.drone_list[id].whole_map[3]
    # 接下来，我需要找到所有的边界点，其特点为，周围的八个临界点里至少有两个的 value 为 0
    # print("map start point", env.drone_list[id].whole_map[1][start_pos[0]][start_pos[1]])
    frontier_point, goal_r, goal_c = find_nearest_frontier(map, start_pos)

    # plt.show()
    # time.sleep(10)
    path, action_list = rescue_path(map=env.joint_map[1],
                                    obstacle_map=env.joint_map[2], target_x=frontier_point[0]\
                                    , target_y=frontier_point[1], start_x=start_pos[0], start_y=start_pos[1])
    # path, action_list = rescue_path(map=free_zone,
    #                                 obstacle_map=obstacle_map, target_x=frontier_point[0] \
    #                                 , target_y=frontier_point[1], start_x=start_pos[0], start_y=start_pos[1])
    # 画路径
    # 这里的 zip(*path) 将 path 列表中的 (x, y) 元组拆分为两个列表，一个是所有的 x 坐标，一个是所有的 y 坐标
    # 然后 ax1.plot(y, x) 会在 ax1 上画出这条路径。
    # print("path is", path)

    # x, y = zip(*path)
    # ax1.plot(y, x, c='r', lw=2)
    # print("action list", action_list)
    return path, action_list, goal_r, goal_c

def rescue_path(map, obstacle_map, target_x, target_y, start_x, start_y):
    astar = rescue.AStar(obstacle_map=obstacle_map, map=map, target_x=target_x, target_y=target_y, start_x=start_x,
                         start_y=start_y)
    return astar.RunAndSaveImage()


