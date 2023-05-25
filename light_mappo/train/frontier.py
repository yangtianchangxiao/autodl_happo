import torch
# from env_Drones import EnvDrones
import random
import sys

sys.path.append(r"d:/code/rl_local/RL-in-multi-robot-foraging/marl")
from light_mappo.config import get_config
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
import time
import numpy as np
from queue import PriorityQueue

# from ddpg_cnn import Actor_CNN
import light_mappo.envs.EnvDrone.classic_control.env_Drones2 as search_grid
from light_mappo.algorithms.algorithm.r_mappo import RMAPPO as TrainAlgo
from light_mappo.algorithms.algorithm.rMAPPOPolicy import RMAPPOPolicy as Policy
import gym
from light_mappo.algorithms.algorithm.r_actor_critic import R_Actor as actor

import sys
import torch.nn.functional as F
import rescue

class rescue_action():
    def __init__(self, actions, id):
        self.actions = actions
        self.id = id
class rescue_astar():
    def __init__(self, agent_repetition, agent_num):
        self.agent_repetition = agent_repetition
        # 保存rescue的动作
        self.resuce_action_list = []
        self.grid_agents = []
        self.last_grid_agents = np.zeros(agent_num)
        self.agent_repetition = np.zeros(agent_num)
    # This function is for find the frontier points when an agent continues failing in exploring new areas.
    def find_frontier(self, current_x, current_y, map):
        # 假设检测点的坐标是[0, 1]
        x, y = current_x, current_y

        # 获取检测点周围的8个点的值
        neighbors = map[max(x - 1, 0):min(x + 2, map.shape[0]), max(y - 1, 0):min(y + 2, map.shape[1])]

        # 判断有几个邻居点的值是0
        num_zeros = np.count_nonzero(neighbors == 0)

        # 如果num_zeros >= 2，则说明有超过两个邻居点的值是0
        if num_zeros >= 2:
            print("存在超过两个的值为0的邻居点")
        else:
            print("不存在超过两个的值为0的邻居点")

    def rescue_path(self, map, obstacle_map, target_x, target_y, start_x, start_y):
        astar = rescue.AStar(obstacle_map=obstacle_map, map=map, target_x=target_x, target_y=target_y, start_x=start_x,
                             start_y=start_y)
        return astar.RunAndSaveImage()

    def find_nearest_frontier(self, map, start):
        rows, cols = map.shape
        queue = PriorityQueue()
        start = (start[0], start[1])
        queue.put((0, start))
        visited = set()
        visited.add(start)

        while not queue.empty():
            distance, (r, c) = queue.get()
            if map[r, c] == 0:
                # print("map rc",map[r,c])
                # return (r, c)
                return last_final_point
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]:
                nr, nc = r + dr, c + dc
                last_final_point = (r, c)
                if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                    continue
                if (nr, nc) in visited:
                    continue

                if map[nr, nc] <= 1:
                    queue.put((distance + 1, (nr, nc)))
                    visited.add((nr, nc))
        return None

    def generate_path(self, env, id):
        print("agent_repetition", self.agent_repetition)
        start_pos = env.drone_list[id].pos
        # 开始规划路径，第一步找到这个agent的地图里所有的边界点，为此，我们需要一份同时包含障碍物和已探索区域的地图
        # 此时，map中 value>=2 的点就是障碍物，0<value<=1的点就是已经搜索的区域
        map = env.drone_list[id].whole_map[1] + 2 * env.drone_list[id].whole_map[3]
        # 接下来，我需要找到所有的边界点，其特点为，周围的八个临界点里至少有两个的 value 为 0
        frontier_point = self.find_nearest_frontier(map, start_pos)
        path, action_list = self.rescue_path(map=env.drone_list[id].whole_map[1],
                                        obstacle_map=env.drone_list[id].whole_map[3], target_x=frontier_point[0] \
                                        , target_y=frontier_point[1], start_x=start_pos[0], start_y=start_pos[1])
        return action_list

def



for i in range(env.drone_num):
    # id starts from 0.
    resuce_action_list.append(rescue_action(actions=[], id=i))
    grid_agents.append(0)
for i_episode in range(1, num_episodes):
    print(env.MC_iter)
    obs = np.array(obs)
    # print("obs.shape",obs.shape)
    action, _, _, _ = actor_1(obs, rnn_state, mask)  # 每个智能体采用自己的状态做动作
    # print("pre - action is", action)
    action = torch.squeeze(action)
    # print("action is ", action)
    one_hot_action = np.eye(num_class)[action]
    # print("one-hot", one_hot_action)
    # print("obs grid each agent", env.find_grid_count)
    for i in range(env.drone_num):
        grid_agents[i] = env.average_list[i]
    # 分别记录两个agent没有探索出新地方的次数
    print("grid agents",grid_agents)
    for i in range(env.drone_num):
        if last_grid_agents[i] == grid_agents[i]:
            agent_repetition[i] = agent_repetition[i] + 1
        else:
            agent_repetition[i] = 0
    last_grid_agents = grid_agents.copy()

    # 当 存在agent连续 5 次都没有探索到新地方的时候，触发rescue机制
    # 将这个 agent 当前所在位置作为 rescue 的起始位置，之后使用 astar 规划到最近的 frontier 去
    for i, repetition in enumerate(agent_repetition):
        if repetition > reputation_threshold:
            print("agent_id is", i)
            resuce_action_list[i].actions = generate_path(env=env, id=i)
            resuce_action_list[i].id = i
            agent_repetition[i] = -10000
    # generate_path(env=env,agent_repetition=agent_repetition)
    for i in range(env.drone_num):
        if len(resuce_action_list[i].actions)>0:
            # 取首个元素，然后删除，直到所有的元素全部删除，此时即到达A star 的目的地
            one_hot_action[i] = resuce_action_list[i].actions.pop(0)
            if len(resuce_action_list) == 0:
                agent_repetition[i] = 0
