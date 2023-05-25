# 在这个环境里, 智能体一开始不会面临一条路是cutting road 的局面,也就是说, 一开始的几条路都可以遍历全图
# 不会出现一开始的必经之路就只有一条

from typing import Optional
import gym
import heapq
from gym import spaces
from gym.envs.registration import register
from gym.utils import seeding
import numpy as np
import random
import copy
from typing import List, Tuple
from itertools import product
from collections import deque


# from gym.envs.classic_control import utils


class Drones(object):
    def __init__(self, pos, view_range, id, map_size):
        self.id = id
        self.pos = pos
        self.view_range = view_range
        self.area = None
        self.communicate_list = [] # 记录可以通信的名单
        self.relative_pos = []
        self.relative_direction = []
        self.relatvie_coordinate = []
        self.individual_observed_zone = []
        self.observed_obs = []
        self.observed_drone = []
        self.individual_observed_obs = None
        self.unobserved = []
        self.communicate_rate = 0  # 添加了机器人通信频率奖励，使机器人在扩散探索的同时也注意信息的共享
        self.whole_map = np.zeros((2, 2*self.view_range, 2*self.view_range), dtype=np.float32)  # 每个机器人保存一个本地地图
        self.last_whole_map = None
        self.grid_communication = 0
        self.obstacle_communication = 0
        self.coord_per_obs = np.empty((4*view_range**2, 2)) # 记录每次每个agent探索的空白区域的坐标，用于后续惩罚agents在一个step中，过多区域重合的现象
        # self.coord_per_obs_length = 0 # 记录agent一次探索的空白区域的坐标有几个

class Human(object):
    def __init__(self, pos):
        self.pos = pos


class Layout(object):
    def __init__(self, map_size, layout):
        self.map_size = map_size
        self.layout = layout


    def generate_obstacles_and_free_spaces(self, width: int, height: int, obstacle_percentage: float) -> Tuple[
        List[Tuple[int, int]], List[Tuple[int, int]]]:
        if obstacle_percentage < 0 or obstacle_percentage > 1:
            raise ValueError("Obstacle percentage must be between 0 and 1.")

        total_cells = width * height
        total_obstacles = int(total_cells * obstacle_percentage)

        obstacle_coordinates = set()
        available_positions = set((x, y) for x in range(1, width - 1) for y in range(1, height - 1))

        while len(obstacle_coordinates) < total_obstacles and available_positions:
            x, y = random.choice(tuple(available_positions))

            tree_pos = {(x + dx, y + dy) for dx, dy in product(range(-1, 2), repeat=2)}
            obstacle_coordinates.update(tree_pos)
            available_positions -= tree_pos

        free_spaces = set((x, y) for x in range(1, width - 1) for y in range(1, height - 1)) - obstacle_coordinates

        return list(obstacle_coordinates), list(free_spaces)
    def will_create_closed_shape(self, obstacle_coordinates: set, x: int, y: int, width: int, height: int) -> bool:
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        visited = set()
        queue = [(x, y)]
        visited.add((x, y))

        while queue:
            current_x, current_y = queue.pop(0)

            for dx, dy in directions:
                next_x, next_y = current_x + dx, current_y + dy

                if 0 <= next_x < width and 0 <= next_y < height and (next_x, next_y) not in visited and (
                next_x, next_y) not in obstacle_coordinates:
                    queue.append((next_x, next_y))
                    visited.add((next_x, next_y))

        return len(visited) + len(obstacle_coordinates) < width * height



    def wall(self):
        if self.map_size == 50:
            # ---训练环境---#
            if self.layout == 'four long wall':  # simple_env_1
                walls = [[14, 14], [14, 17], [36, 36], [36, 33], [14, 36], [14, 33],
                         [36, 14], [36, 17]]
            if self.layout == 'inverse wall':  # simple_env_2
                walls = [[16, 14], [16, 17], [13, 17], [10, 17], [10, 14],
                         [34, 36], [34, 33], [37, 33], [40, 33], [40, 36],
                         [16, 36], [16, 33], [13, 33], [10, 33], [10, 36],
                         [34, 14], [34, 17], [37, 17], [40, 17], [40, 14]]
            if self.layout == 'moderate_1_train':
                walls = [[8, 8], [8, 11], [8, 14], [8, 16], [11, 8], [14, 8],
                         [17, 8], [20, 8], [23, 8], [25, 8], [25, 11], [25, 14], [25, 17],
                         [25, 20], [25, 23], [25, 23], [25, 26], [25, 29], [25, 32], [25, 35],
                         [25, 38], [25, 41], [26, 41], [29, 41], [32, 41], [35, 41],
                         [38, 41], [41, 41], [41, 38], [41, 35], [41, 33], [41, 32],
                         [41, 24], [38, 24], [35, 24], [32, 24], [29, 24], [26, 24], [23, 24],
                         [20, 24], [17, 24], [14, 24], [11, 24], [8, 24], [41, 22], [41, 19],
                         [41, 16], [41, 13], [41, 10], [41, 8], [38, 8], [35, 8], [33, 8],
                         [8, 27], [8, 30], [8, 33], [8, 36], [8, 39], [8, 41], [11, 41],
                         [14, 41], [17, 41]]
            if self.layout == 'moderate_2_train':
                walls = [[8, 8], [8, 11], [8, 14], [8, 17], [8, 20], [8, 21], [11, 21],
                         [14, 21], [17, 21], [20, 21], [21, 21], [21, 18], [21, 15],
                         [21, 12], [21, 9], [21, 8], [18, 8], [11, 8], [8, 29], [8, 31],
                         [8, 39], [8, 41], [11, 41], [14, 41], [17, 41], [20, 41], [21, 41],
                         [21, 38], [21, 35], [21, 32], [21, 29], [18, 29], [15, 29], [12, 29],
                         [10, 29], [29, 8], [29, 11], [29, 14], [29, 17], [29, 20], [29, 21],
                         [32, 21], [35, 21], [38, 21], [41, 21], [41, 18], [41, 10], [41, 8],
                         [38, 8], [35, 8], [32, 8], [29, 8], [29, 29], [29, 32], [29, 35],
                         [29, 38], [29, 41], [31, 41], [39, 41], [41, 41], [41, 38], [41, 35],
                         [41, 32], [41, 29], [38, 29], [35, 29], [32, 29]]
            if self.layout == 'indoor':  # complex_env_1
                walls = [[8, 8], [8, 11], [8, 14], [11, 14], [11, 17], [17, 17],
                         [17, 14], [20, 14], [20, 11], [20, 8], [11, 20], [11, 23],
                         [11, 26], [11, 29], [11, 32], [8, 32], [8, 35], [8, 38], [8, 40],
                         [17, 23], [17, 20], [17, 29], [17, 32], [20, 32], [20, 35],
                         [20, 38], [20, 40], [20, 23], [23, 23], [26, 23], [29, 23],
                         [29, 20], [29, 17], [29, 14], [29, 8], [29, 5], [29, 2],
                         [32, 14], [35, 14], [38, 14], [41, 14], [41, 17], [41, 20],
                         [41, 23], [41, 26], [44, 26], [47, 26], [41, 29], [41, 32],
                         [41, 35], [41, 38], [38, 38], [35, 38], [38, 26], [35, 26],
                         [32, 38], [32, 41], [32, 44], [47, 38]]
            # ---测试环境---#
            if self.layout == 'moderate_test':
                walls = [[2, 20], [5, 20], [8, 20], [11, 20], [14, 20],
                         [17, 20], [20, 20], [20, 17], [20, 14], [20, 11], [20, 8],
                         [20, 30], [20, 32], [20, 35], [20, 38], [20, 41], [20, 44],
                         [20, 47], [17, 30], [14, 30], [11, 30], [8, 30], [29, 20],
                         [32, 20], [35, 20], [38, 20], [41, 20], [29, 17],
                         [29, 14], [29, 11], [29, 8], [29, 5], [29, 2], [29, 30],
                         [29, 32], [29, 35], [29, 38], [29, 41], [32, 30],
                         [35, 30], [38, 30], [41, 30], [44, 30], [47, 30]]
            if self.layout == 'hard_test':
                walls = [[2, 11], [5, 11], [8, 11], [11, 11], [11, 8], [8, 19], [11, 19],
                         [11, 22], [11, 25], [11, 28], [11, 30], [8, 30], [5, 30], [2, 30],
                         [8, 38], [11, 38], [11, 41], [11, 44], [11, 47], [11, 48], [19, 2],
                         [19, 5], [19, 8], [19, 11], [22, 11], [25, 11], [28, 11], [30, 11],
                         [30, 8], [19, 19], [21, 19], [28, 19], [30, 19], [30, 21], [30, 28],
                         [30, 30], [28, 30], [21, 30], [19, 30], [19, 28], [19, 21], [38, 19],
                         [41, 19], [44, 19], [47, 19], [48, 19], [38, 22], [38, 25], [38, 28],
                         [38, 30], [41, 30], [30, 38], [30, 41], [30, 38], [30, 41], [30, 44],
                         [30, 47], [30, 48], [27, 38], [24, 38], [21, 38], [19, 38], [19, 41],
                         [38, 38], [41, 38], [44, 38], [47, 38], [48, 38], [38, 41], [38, 2],
                         [38, 5], [38, 8], [38, 11], [41, 11]]

            return walls


class SearchGrid(gym.Env):
    def __init__(self):
        # self.observation_space = spaces.Box(low=0, high=1, shape=(4, 50, 50))
        # self.action_space = spaces.Discrete(4)
        # 注意，当我用evaluate的时候，使用的是下面两行，当我运行train的时候，暂时使用的是上面两行，下面两行是否可行，暂时没有测试
        # 补充，经过测试，发现似乎确实可行，那么就暂时决定grid_drone就这么用了
        # When use mlp
        self.view_range = 10
        # self.view_range = 6
        self.observation_space = spaces.Box(low=0, high=1, shape=(2 * self.view_range * self.view_range,))
        self.share_observation_space = spaces.Box(low=0, high=1, shape=(3 * 50* 50,))
        # When use cnn
        # self.observation_space = spaces.Box(low=0, high=1, shape=(4, 50, 50,))
        # self.share_observation_space = spaces.Box(low=0, high=1, shape=(3, 50, 50,))
        # self.observation_map_space = spaces.Box(low=)
        self.action_space = spaces.Discrete(4)
        self.init_param()
        # print("share_observation_space",self.share_observation_space)
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # print("目标点总数",self.human_num)
        self.MC_iter = self.MC_iter + 1
        # print("McITER",self.MC_iter)
        # print("action is",action)
        self.drone_step(action)
        self.human_take_action()
        self.human_step(self.human_act_list)
        # self.get_full_obs()
        self.get_joint_obs(self.MC_iter)
        observation, reward, done, info = self.state_action_reward_done()
        # # 是 使用 individual reward or shared reward
        #
        # reward = np.full_like(reward, np.mean(reward) * len(reward))  # 使用numpy的广播功能对所有奖励进行均值填充
        reward = np.full_like(reward, np.mean(reward))  # 使用numpy的广播功能对所有奖励进行均值填充
        observation = [o.flatten() for o in observation]  # 使用列表推导式对每个观测值进行扁平化处理

        return observation, reward, done, info, self.joint_map.ravel()

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False,
              options: Optional[dict] = None):
        self.init_param()
        self.MC_iter += 1
        self.get_joint_obs(self.MC_iter)
        self.last_drone_pos += [drone.pos for drone in self.drone_list]
        observation, _, _, info = self.state_action_reward_done()

        # 使用列表推导式对每个观测值进行扁平化处理
        observation = [o.flatten() for o in observation]

        # 当使用mlp时，才需要将joint_map flatten，否则是不需要的
        # print("!!!!!!!!shape of joint map",self.joint_map.shape)
        return observation, self.joint_map.flatten()
        # return (observation[0])

    def render(self):
        pass

    def close(self):
        pass

    def drone_step(self, drone_act_list):
        # 定义每个方向的增量
        delta = [np.array([-1, 0]), np.array([1, 0]), np.array([0, -1]), np.array([0, 1])]

        # 更新每个机器人的位置
        self.collision = np.zeros(self.drone_num)
        for k in range(self.drone_num):
            # 获取机器人要执行的操作
            # 当评价多机时，用这个
            action = drone_act_list[k]
            # 当评价单机时，用这个。但是训练的时候，还是用上面的
            # action = drone_act_list

            if max(action) == 0:
                continue
            # 根据操作计算出机器人的新位置
            direction = np.argmax(action)  # 获取机器人要移动的方向

            temp_pos = self.drone_list[k].pos + delta[direction]  # 根据方向更新机器人的位置
            # 禁止撞击
            # if self.land_mark_map[temp_pos[0], temp_pos[1]] > 0:
            #     self.collision[k] = 1
            # else:
            #     self.drone_list[k].pos = temp_pos

            # 不禁止撞击
            self.drone_list[k].pos = temp_pos

    def human_take_action(self):
        self.human_act_list = [0] * self.human_num
        for i in range(self.human_num):
            self.human_act_list[i] = random.randint(0, 3)

    # def human_step(self, human_act_list):
    #     for k in range(self.human_num):
    #         # print(self.human_init_pos)
    #         # print([self.human_list[k].pos[0]-self.human_init_pos[k][0], self.human_list[k].pos[1]-self.human_init_pos[k][1]])
    #         if human_act_list[k] == 0:
    #             if self.human_list[k].pos[0] > 0 and (self.human_list[k].pos[0] - \
    #                                                   self.human_init_pos[k][0] - 1 > -self.move_threshold):
    #                 free_space = self.land_mark_map[self.human_list[k].pos[0] - 1, self.human_list[k].pos[1]]
    #                 if free_space == 0:
    #                     self.human_list[k].pos[0] = self.human_list[k].pos[0] - 1
    #         elif human_act_list[k] == 1:
    #             if self.human_list[k].pos[0] < self.map_size - 1 and (self.human_list[k].pos[0] - \
    #                                                                   self.human_init_pos[k][
    #                                                                       0] + 1 < self.move_threshold):
    #                 free_space = self.land_mark_map[self.human_list[k].pos[0] + 1, self.human_list[k].pos[1]]
    #                 if free_space == 0:
    #                     self.human_list[k].pos[0] = self.human_list[k].pos[0] + 1
    #         elif human_act_list[k] == 2:
    #             if self.human_list[k].pos[1] > 0 and (self.human_list[k].pos[1] - \
    #                                                   self.human_init_pos[k][1] - 1 > -self.move_threshold):
    #                 free_space = self.land_mark_map[self.human_list[k].pos[0], self.human_list[k].pos[1] - 1]
    #                 if free_space == 0:
    #                     self.human_list[k].pos[1] = self.human_list[k].pos[1] - 1
    #         elif human_act_list[k] == 3:
    #             if self.human_list[k].pos[1] < self.map_size - 1 and (self.human_list[k].pos[1] - \
    #                                                                   self.human_init_pos[k][
    #                                                                       1] + 1 < self.move_threshold):
    #                 free_space = self.land_mark_map[self.human_list[k].pos[0], self.human_list[k].pos[1] + 1]
    #                 if free_space == 0:
    #                     self.human_list[k].pos[1] = self.human_list[k].pos[1] + 1

    def human_step(self, human_act_list):
        for k in range(self.human_num):
            human_pos = self.human_list[k].pos
            human_init_pos = self.human_init_pos[k]

            if human_act_list[k] == 0:
                new_pos = human_pos[0] - 1
                if new_pos > 0 and new_pos - human_init_pos[0] > -self.move_threshold:
                    free_space = self.land_mark_map[new_pos, human_pos[1]]
                    if free_space == 0:
                        human_pos[0] = new_pos
            elif human_act_list[k] == 1:
                new_pos = human_pos[0] + 1
                if new_pos < self.map_size - 1 and new_pos - human_init_pos[0] < self.move_threshold:
                    free_space = self.land_mark_map[new_pos, human_pos[1]]
                    if free_space == 0:
                        human_pos[0] = new_pos
            elif human_act_list[k] == 2:
                new_pos = human_pos[1] - 1
                if new_pos > 0 and new_pos - human_init_pos[1] > -self.move_threshold:
                    free_space = self.land_mark_map[human_pos[0], new_pos]
                    if free_space == 0:
                        human_pos[1] = new_pos
            elif human_act_list[k] == 3:
                new_pos = human_pos[1] + 1
                if new_pos < self.map_size - 1 and new_pos - human_init_pos[1] < self.move_threshold:
                    free_space = self.land_mark_map[human_pos[0], new_pos]
                    if free_space == 0:
                        human_pos[1] = new_pos


    def get_full_obs(self):
        # Initialize an array with ones
        obs = np.ones((self.map_size, self.map_size, 3))

        # Set [0, 0, 0] for wall and tree locations
        wall_tree_mask = np.logical_or(self.land_mark_map == 1, self.land_mark_map == 2)
        obs[wall_tree_mask] = 0

        # Set [1, 0, 0] for human locations
        for i in range(self.human_num):
            human_pos = tuple(self.human_list[i].pos)
            obs[human_pos] = [1, 0, 0]

        # Set [0.5*i, 0, 0.5*i] for drone locations
        for i in range(self.drone_num):
            drone_pos = tuple(self.drone_list[i].pos)
            obs[drone_pos] = [0.5 * i, 0, 0.5 * i]

        return obs

    def get_drone_obs(self, drone):  # 获得无人机的观测，这里的drone是类
        drone.observed_obs = []
        drone.unobserved = []
        drone.individual_observed_obs = 0
        drone.observed_drone = []
        drone.communicate_rate = 0
        drone.last_whole_map = drone.whole_map.copy()
        index = random.randint(self.sensing_threshold[0], self.sensing_threshold[1])
        obs_size = 2 * drone.view_range - 1
        sensing_size = 2 * (drone.view_range + index) - 1
        obs = np.ones((obs_size, obs_size, 3))
        # 这里是给机器人感知其他机器人的位置加了波动

        # 对于单个agent，第一层记录自己的观测
        # 第二层记录自己对障碍物的观测

        #对于joint obs，是 obs 的 joint整合



        # 先通过通信更新得图：

        # 在观测范围内进行信息更新，更新时间戳地图1，轨迹地图2和障碍物地图3
        # 禁止通信
        for k in range(self.drone_num):
            if self.drone_list[k].id != drone.id and (self.drone_list[k].pos[0] - drone.pos[0]) ** 2 \
                    + (self.drone_list[k].pos[1] - drone.pos[1]) ** 2 <= sensing_size ** 2:
                drone.communicate_list.append(k)
                # 记录其他智能体的位置，以后可以变成记录其他智能体的轨迹
                drone.whole_map[2, self.drone_list[k].pos[0]-drone.pos[0]+self.view_range, self.drone_list[k].pos[1]-drone.pos[1] + ] = 1
                # print("信息交换")
                # print("距离的平方是", (self.drone_list[k].pos[0]-drone.pos[0])**2\
                #     + (self.drone_list[k].pos[1]-drone.pos[1])**2)
                # print("距离阈值是",sensing_size**2)
        drone.grid_communication = 0
        drone.obstacle_communication = 0
        # if drone.communicate_list:
        #     maps = np.array([self.drone_list[i].whole_map for i in drone.communicate_list] + [drone.whole_map])
        #     max_2 = np.max(maps[:, 2, :, :], axis=0)
        #     max_1 = np.max(maps[:, 1, :, :], axis=0)
        #     max_3 = np.max(maps[:, 3, :, :], axis=0)
        #
        #     update_3 = max_3 > drone.whole_map[3, :, :]
        #     # update_1 = max_1 > drone.whole_map[1, :, :]
        #
        #     drone.grid_communication += np.count_nonzero(max_1) - np.count_nonzero(drone.whole_map[1])
        #     drone.obstacle_communication += np.sum(update_3)
        #     # print("drone.grid_communication",drone.grid_communication)
        #     # print("drone.obstacle_communication",drone.obstacle_communication)
        #
        #     drone.whole_map[2, :, :] = np.maximum(max_2, drone.whole_map[2, :, :])
        #     drone.whole_map[3, :, :] = max_3
        #     drone.whole_map[1, :, :] = max_1

        if drone.communicate_list:
            # Combine maps from all drones, including the current one
            maps = np.array([self.drone_list[i].whole_map for i in drone.communicate_list] + [drone.whole_map])

            # Compute the maximum values for channels 1, 2, and 3
            max_channels = np.max(maps[:, [1, 2, 3], :, :], axis=0)

            # Update grid and obstacle communication values
            drone.grid_communication += np.count_nonzero(max_channels[0]) - np.count_nonzero(drone.whole_map[1])
            drone.obstacle_communication += np.sum(max_channels[2] > drone.whole_map[3, :, :])

            # Update the drone's whole_map with the maximum values
            drone.whole_map[[1, 2, 3], :, :] = max_channels

        # Large map
        drone.whole_map[0] = 0
        drone.whole_map[0, :drone.pos[0]+1, :drone.pos[1]+1] = self.memory_step

        # drone.whole_map[0, drone.pos[0], drone.pos[1]] = self.memory_step
        # for i in range(sensing_size):
        #     for j in range(sensing_size):
        #         x = i + drone.pos[0] - (drone.view_range + index) + 1
        #         y = j + drone.pos[1] - (drone.view_range + index) + 1
        #         for k in range(self.drone_num):  # 是否有其他机器人在观测范围内
        #             if self.drone_list[k].pos[0] == x and self.drone_list[k].pos[1] == y:
        #                 if self.drone_list[k].id != drone.id:
        #                     drone.observed_drone.append([x, y])
        #                     drone.whole_map[
        #                         2, x, y] = self.memory_step  # add other agent's history positions to the map
        #                     drone.communicate_rate += 1

        # 确定观测到的区域, 代替上面注释掉的部分
        x_indices = np.arange(drone.pos[0] - (drone.view_range + index) + 1, drone.pos[0] + drone.view_range - index)
        y_indices = np.arange(drone.pos[1] - (drone.view_range + index) + 1, drone.pos[1] + drone.view_range - index)
        # 确定在观察区域内是否有其他无人机，以及在哪里
        for other_drone in self.drone_list:
            if other_drone.id != drone.id and other_drone.pos[0] in x_indices and other_drone.pos[1] in y_indices:
                drone.observed_drone.append([other_drone.pos[0], other_drone.pos[1]])
                drone.whole_map[
                    2, other_drone.pos[0], other_drone.pos[
                        1]] = self.memory_step  # add other agent's history positions to the map
                drone.communicate_rate += 1

        # 这里循环的目的是构建障碍物地图
        # Compute the range of x and y coordinates
        x_range = range(drone.pos[0] - drone.view_range + 1, drone.pos[0] + drone.view_range)
        y_range = range(drone.pos[1] - drone.view_range + 1, drone.pos[1] + drone.view_range)

        # # Clip x_range and y_range to the map boundaries
        # x_range = [x for x in x_range if 0 <= x <= self.map_size - 1]
        # y_range = [y for y in y_range if 0 <= y <= self.map_size - 1]
        #
        # # Iterate over the clipped x_range and y_range
        # for x, i in zip(x_range, range(obs_size)):
        #     for y, j in zip(y_range, range(obs_size)):
        #         if self.land_mark_map[x, y] == 2:
        #             obs[i, j] = 0

        # Create a meshgrid with the correct dimensions
        xx, yy = np.meshgrid(np.arange(obs_size), np.arange(obs_size), indexing='ij')

        # Calculate the actual x and y coordinates in the land_mark_map
        actual_x = np.clip(drone.pos[0] - obs_size // 2 + xx, 0, self.map_size - 1)
        actual_y = np.clip(drone.pos[1] - obs_size // 2 + yy, 0, self.map_size - 1)

        # Find the indices in the land_mark_map where the value is 2
        mask = (self.land_mark_map[actual_x, actual_y] == 2)

        # Set the corresponding indices in the obs array to 0
        obs[mask] = 0

        # # Create a meshgrid with the correct dimensions
        # xx, yy = np.meshgrid(np.arange(obs_size), np.arange(obs_size), indexing='ij')
        #
        # # Calculate the actual x and y coordinates in the land_mark_map
        # actual_x = np.clip(drone.pos[0] - obs_size + xx, 0, self.map_size - 1)
        # actual_y = np.clip(drone.pos[1] - obs_size + yy, 0, self.map_size - 1)
        #
        # # Find the indices in the land_mark_map where the value is 2
        # mask = (self.land_mark_map[actual_x, actual_y] == 2)
        #
        # # Set the corresponding indices in the obs array to 0
        # obs[mask] = 0

        coord_per_obs_length = 0 # 记录agent一次探索的空白区域的坐标有几个
        drone.coord_per_obs = np.empty((obs_size**2, 2)) # 记录每次每个agent探索的空白区域的坐标，用于后续惩罚agents在一个step中，过多区域重合的现象
        for i in range(obs_size):
            for j in range(obs_size):
                x = i + drone.pos[0] - drone.view_range + 1
                y = j + drone.pos[1] - drone.view_range + 1
                # if 0 <= x < 50 and 0 <= y < 50:
                #     drone.whole_map[1, x, y] = 1  # Add cell's timestamp to an agent's whole map.
                for k in range(self.human_num):  # 是否有目标点在观测范围内
                    if self.human_list[k].pos[0] == x and self.human_list[k].pos[1] == y:
                        obs[i, j, 0] = 1
                        obs[i, j, 1] = 0
                        obs[i, j, 2] = 0
                for k in range(self.drone_num):  # 是否有其他机器人在观测范围内
                    if self.drone_list[k].pos[0] == x and self.drone_list[k].pos[1] == y:
                        obs[i, j, 0] = 0.5
                        obs[i, j, 1] = 0
                        obs[i, j, 2] = 0.5
                if 0 <= x <= self.map_size - 1 and 0 <= y <= self.map_size - 1:  # 是否有障碍物在观测范围内
                    if self.land_mark_map[x, y] == 1:
                        obs[i, j] = 0
                    if self.land_mark_map[x, y] == 2:  # 在发现障碍物后对观测进行处理
                        obs[i, j] = 0
                        drone.observed_obs.append([x, y])
                        gap = [drone.observed_obs[-1][0] - drone.pos[0], \
                               drone.observed_obs[-1][1] - drone.pos[1]]
                        gap_abs = [abs(drone.observed_obs[-1][0] - drone.pos[0]), \
                                   abs(drone.observed_obs[-1][1] - drone.pos[1])]
                        chosen_gap = max(gap_abs)
                        directions = [(1, 1), (-1, 1), (1, -1), (-1, -1)]

                        for dx, dy in directions:
                            if 0 <= gap[0] * dx <= drone.view_range and 0 <= gap[1] * dy <= drone.view_range:
                                if gap[0] == 0:
                                    for num in range(drone.view_range - chosen_gap - 1):
                                        for num_1 in range(num + 2):
                                            drone.unobserved.append([i + dx * num_1, j + dy * (num + 1)])
                                else:
                                    for num in range(drone.view_range - chosen_gap - 1):
                                        drone.unobserved.append([i + dx * (num + 1), j + dy * (num + 1)])

                    else:
                        # 对观测到的区域添加时间戳
                        drone.whole_map[1, x, y] = 1
                        # print("droe.coord_per_obs shape",drone.coord_per_obs.shape)
                        drone.coord_per_obs[coord_per_obs_length] = (x, y)
                        coord_per_obs_length = coord_per_obs_length + 1

                else:  # 其他情况
                    obs[i, j] = 0.5

                # 这里是设置圆形观测区域
                if (drone.view_range - 1 - i) * (drone.view_range - 1 - i) + (drone.view_range - 1 - j) * (
                        drone.view_range - 1 - j) > drone.view_range * drone.view_range:
                    obs[i, j] = 0.5
        drone.coord_per_obs = drone.coord_per_obs[:coord_per_obs_length]
        for pos in drone.unobserved:  # 这里处理后得到的obs是能观测到的标志物地图
            obs[pos[0], pos[1]] = 0.5

        # 进行轨迹的衰减, 是上面那段注释的优化
        drone.whole_map[2, :, :] -= 1 / self.t_u
        drone.whole_map[2, drone.whole_map[2, :, :] < 1 / self.t_u] = 0
        # 进行自身探索地图的衰减
        drone.whole_map[1, :, :] = np.maximum(0, drone.whole_map[1, :, :] - 0.0025)

        # print("relative_direction:",drone.relative_direction)



        return obs

    def get_joint_obs(self, time_stamp):
        # Modification record which agent find the target
        # One target can be found by multi agents at the smae time point.
        # Target_per_agent defines how many targets are found by each agent at thie time point.
        self.target_per_agent = np.zeros(self.drone_num)
        human_del_list = []
        len_human_del_list = 0
        # Record obstacle gain of each agent
        # self.obstacle_gain_per_agent = np.zeros(self.drone_num)
        len_obstacle_gain = len(self.obstacles)

        # self.obstacles_temp = copy.deepcopy(self.obstacles)
        self.per_observed_goal_num = 0
        self.time_stamp = time_stamp
        obs = np.full((self.map_size, self.map_size, 3), 0.5)
        # self.obstacle_multi_agent = [self.obstacles for k in range(self.drone_num)]
        # 打乱智能体的决策顺序，随机抽取智能体来决策
        allowed_values = list(range(self.drone_num))
        k_list = random.sample(allowed_values, self.drone_num)
        for k in k_list:
            self.drone_list[k].individual_observed_obs = 0
            temp = self.get_drone_obs(self.drone_list[k])
            size = temp.shape[0]
            temp_list_individual = []

            for i in range(size):
                for j in range(size):
                    x = i + self.drone_list[k].pos[0] - self.drone_list[k].view_range + 1
                    y = j + self.drone_list[k].pos[1] - self.drone_list[k].view_range + 1

                    # 如果一个位置根本没有被观测到，就不执行赋值
                    if np.all(temp[i, j] == (0.5,0.5,0.5)):
                        continue
                    else:
                        obs[x, y] = temp[i, j]
                        temp_list_individual.append([x, y])
                        # 这里为了判断观测中有多少障碍物，并更新障碍物地图
                        # 如果temp[i,j] = (0,0,0)
                        if not temp[i, j].any():
                            if [x, y] not in self.obstacles:
                                self.drone_list[k].individual_observed_obs += 1
                                # self.obstacle_multi_agent[k].append([x,y])
                                self.obstacles.append([x, y])  # 所有机器人观测过的障碍物
                            self.drone_list[k].whole_map[
                                3, x, y] = 1  # add obstacle information to each agent's whole map
                        # 如果观测中有目标，则清除被观测到的目标
                        if all(obs[x, y] == [1, 0, 0]):
                            self.per_observed_goal_num += 1
                            for num, goal in enumerate(self.human_list):
                                if goal.pos[0] == x and goal.pos[1] == y:
                                    human_del_list.append(num)
                                    # print("11111111111111111111")


            self.obstacle_gain_per_agent[k] = len(self.obstacles) - len_obstacle_gain
            len_obstacle_gain = len(self.obstacles)
            self.target_per_agent[k] = self.target_per_agent[k] + len(human_del_list) - len_human_del_list
            # print("self.target_per_agent:",k,self.target_per_agent[k])
            len_human_del_list = len(human_del_list)
            # self.drone_list[k].individual_observed_zone = temp_list_individual
            # 这里计算观测区域去掉障碍物的面积
            # self.drone_list[k].area = len(self.drone_list[k].individual_observed_zone) - \
            #                           self.drone_list[k].individual_observed_obs
            # print(len(self.drone_list[k].individual_observed_zone))
        # Delete all targets found at this time point
        # 去掉重复检测到的target
        # human_del_list = list(set(human_del_list))
        # if len(human_del_list) > 0:
        #     # print("1111111")
        #     new_human_list = []
        #     new_human_init_pos = []
        #     for i in range(len(self.human_list)):
        #         if i in human_del_list:
        #             self.human_num -= 1
        #         else:
        #             new_human_list.append(self.human_list[i])
        #             new_human_init_pos.append(self.human_init_pos[i])
        #     self.human_list = new_human_list
        #     self.human_init_pos = new_human_init_pos

        # 去掉重复检测到的target
        # Convert human_del_list to a set to remove duplicates
        human_del_set = set(human_del_list)
        # Use list comprehension to create new_human_list and new_human_init_pos
        new_human_list = [h for i, h in enumerate(self.human_list) if i not in human_del_set]
        new_human_init_pos = [h for i, h in enumerate(self.human_init_pos) if i not in human_del_set]
        # Update human_count and human_list
        self.human_num = len(new_human_list)
        self.human_list = new_human_list
        self.human_init_pos = new_human_init_pos

        # 合并所有无人机的整个地图

        return obs

    def state_action_reward_done(self):  # 这里返回状态值，奖励值，以及游戏是否结束
        # print("reward is")
        # reward = 0  # 合作任务，只设置单一奖励
        # reward_list = np.zeros(self.drone_num, dtype=np.float32)
        ####################设置奖励的增益
        target_factor = 200
        # 发现障碍物的奖励系数
        information_gain = 0.9
        # distance_factor = 0.05
        # pos_without_change_factor = 20
        # time step factor 变成 0, 取消时间惩罚
        # 时间惩罚
        time_step_factor = 1
        # 发现新区域的奖励系数
        average_time_stamp_factor = 0.5
        collision_factor = 500
        # 单步内，智能体探索区域重复的惩罚系数
        overlap_factor = 0.02

        # 对reward进行缩放，降低回报的波动，减少学习的难度。（深度强化学习落地指南 p77.
        reward_scale = 0.1
        # original
        # target_factor = 10
        # information_gain = 0.5
        # distance_factor = 0.005
        # pos_without_change_factor = 20
        # # time step factor 变成 0, 取消时间惩罚
        # time_step_factor = 1
        # average_time_stamp_factor = 0.2
        # collision_factor = 500

        #
        # target_factor = 10
        # information_gain = 0.2
        # distance_factor = 0.05
        # pos_without_change_factor = 20
        # time_step_factor = 6
        # average_time_stamp_factor = 0.2
        # collision_factor = 500

        # target_factor = 10
        # information_gain = 0.1
        # distance_factor = 0.1
        ####################
        # for i in range(self.drone_num):   #这里可以做智能信用分配
        #     reward += self.compute_reward(self.drone_list[i])
        done = False
        single_map_set = [self.drone_list[k].whole_map for k in range(self.drone_num)]

        info = {}
        reward_list = [target_factor * i_agent for i_agent in self.target_per_agent]  # 这里计算发现目标点的数量

        if self.human_num == 0:
            reward_list = list(map(lambda x: x + 500, reward_list))
            done = True
            # info['0'] = "find all target"
        for i in range(self.drone_num - 1):  # 如果机器人发生碰撞
            for j in range(i + 1, self.drone_num):
                distance = np.linalg.norm(np.array(self.drone_list[i].pos) - np.array(self.drone_list[j].pos))
                if distance <= 1:
                    done = True
                    reward_list[i] -= collision_factor
                    reward_list[j] -= collision_factor

        for i, drone in enumerate(self.drone_list):
            if self.land_mark_map[drone.pos[0], drone.pos[1]] > 0:
                # # 如果碰撞, 使无人机位置和上一轮一样,从而保持不动
                done = True
                reward_list[drone.id] -= collision_factor

        if self.time_stamp > self.run_time:  # 超时
            done = False
            reward_list = list(map(lambda x: x - 100, reward_list))
            # print("Time out!")
            # info['0'] = "exceed run time"
        if len(self.obstacles) == self.global_obs_num:
            done = False
            reward_list = list(map(lambda x: x + 500, reward_list))
            # info['0'] = "construct the feature map"
            # print("find all obstacles", 500)
        reward_list = list(map(lambda x: x - time_step_factor, reward_list))  # 单步惩罚
        # print("reward_list time", reward_list)

        for i in range(self.drone_num):
            # self.obstacle_gain_per_agent[i] = np.count_nonzero(self.drone_list[i].whole_map[3])
            reward_list[i] = reward_list[i] + information_gain * self.obstacle_gain_per_agent[i]
            # reward_list[i] = reward_list[i] + information_gain * (
            #             self.obstacle_gain_per_agent[i] - self.last_obstacle_gain_per_agent[i] - self.drone_list[
            #         i].obstacle_communication)
            # print("agent", i, "information gain", information_gain * (self.obstacle_gain_per_agent[i] - self.last_obstacle_gain_per_agent[i]-self.drone_list[i].obstacle_communication))
            # self.last_obstacle_gain_per_agent[i] = self.obstacle_gain_per_agent[i]
        # dis = 0
        # print("reward_list obstacle", reward_list)

        # for i in range(self.drone_num):
        #     dis += ((self.drone_list[i].pos[0] - self.drone_list[0].pos[0]) ** 2 + \
        #             (self.drone_list[i].pos[1] - self.drone_list[0].pos[1]) ** 2) ** 0.5
        # reward += distance_factor * dis
        # print("distance_factor * dis",distance_factor * dis)
        # reward_list = list(map(lambda x: x + distance_factor * dis, reward_list))

        # 机器人的每个时间戳平均尽可能大，保证尽可能有多的区域被探索到
        for i, single_map in enumerate(single_map_set):
            # print("np.sum(single_map[1])",np.sum(single_map[1]))
            # print("非零数数量",np.count_nonzero(single_map[1]))
            self.find_grid_count[i] = np.count_nonzero(single_map[1])
            # average_list[i] = np.sum(single_map[1]) / self.map_size ** 2
            # print("self.find_grid_count[i] and self.last_find_grid_cout[i]", self.find_grid_count[i], self.last_find_grid_cout[i])
            self.average_list[i] = self.find_grid_count[i] - self.last_find_grid_cout[i] - self.drone_list[
                i].grid_communication
            # print("average_list", i, "is", self.average_list[i])
            # print("agent",i,"new area reward",average_list[i] * average_time_stamp_factor)
            self.last_find_grid_cout[i] = self.find_grid_count[i]
        # 如果存在lazy agent，那么将不能得到奖励
        if any(i == 0 for i in self.average_list):
            self.average_list = [0] * self.drone_num
        # print("new area reward",np.arraybd(average_list) * average_time_stamp_factor)
        reward_list = [x + y * average_time_stamp_factor for x, y in zip(reward_list, self.average_list)]
        # print("reward_list new area", reward_list)

        # 单步内探索区域重叠的惩罚：
        # 将所有坐标堆叠到一个数组中
        all_coordinates = np.vstack([self.drone_list[i].coord_per_obs for i in range(self.drone_num)])

        # 使用 np.unique 函数找到唯一的坐标
        unique_coordinates, counts = np.unique(all_coordinates, axis=0, return_counts=True)

        # 计算重复坐标的数量
        num_duplicates = np.sum(counts > 1)
        reward_list = list(map(lambda x: reward_scale * (x - num_duplicates*overlap_factor), reward_list))  # 单步惩罚
        # average = sum(average_list) / self.drone_num * average_time_stamp_factor
        # print("average is",average)
        done_list = [done for i_agent in range(self.drone_num)]
        # reward_list = list(map(lambda x: x+average, reward_list))
        # print("reward list is",reward_list)
        target_found_num = self.human_num_copy - self.human_num
        if done is True or self.time_stamp > self.run_time:
            # print("self.human_num_copy-self.human_num", target_found_num)
            self.reset()
        return single_map_set, reward_list, done_list, target_found_num

    def compute_reward(self, drone):  # s->a->r->s'
        pos_factor = 0.2

        direction_factor = 0.01
        target_factor = 300
        communicate_factor = 10
        time_factor = 1
        information_gain = 5 / 3
        # sum_1 = 0
        reward = 0


        return reward

    def get_neighboring_free_spaces(self, pos: Tuple[int, int], free_spaces: List[Tuple[int, int]], distance: int) -> \
        List[Tuple[int, int]]:

        def is_valid_neighbor(free_space):
            x_diff = abs(pos[0] - free_space[0])
            y_diff = abs(pos[1] - free_space[1])
            neigobor_distance = np.sqrt(x_diff ** 2 + y_diff ** 2)
            return x_diff < distance and y_diff < distance and neigobor_distance > 1

        random.shuffle(free_spaces)
        neighbors = [free_space for free_space in free_spaces if is_valid_neighbor(free_space)]

        return neighbors[:self.drone_num - 1]

    def pick_random_positions(self, free_spaces: List[Tuple[int, int]], x: int) -> Tuple[
        Tuple[int, int], List[Tuple[int, int]]]:
        tried_positions = set()
        while len(tried_positions) < len(free_spaces):
            temp_pos = random.choice(free_spaces)
            if temp_pos in tried_positions:
                continue
            tried_positions.add(temp_pos)
            distance = x + 2
            eligible_positions = self.get_neighboring_free_spaces(temp_pos, free_spaces, distance)

            # x 是总智能体数量-1。因为第一个agent已经被选择了，然后基于此，我们选择剩下的智能体
            if len(eligible_positions) >= x:
                return temp_pos, eligible_positions

        raise ValueError("Cannot find a temp_pos with enough eligible positions.")


    def is_valid_coord(self, coord, width, height, obstacle_coordinates):
        x, y = coord
        return 1 <= x < width - 1 and 1 <= y < height - 1 and coord not in obstacle_coordinates

    def get_neighbors(self, coord, width, height, obstacle_coordinates):
        x, y = coord
        return [(x + dx, y + dy) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)] if
                self.is_valid_coord((x + dx, y + dy), width, height, obstacle_coordinates)]

    def heuristic(self, coord, target_coord):
        x1, y1 = coord
        x2, y2 = target_coord
        return abs(x1 - x2) + abs(y1 - y2)

    def is_path_available(self, agent_coord, target_coord, width, height, obstacle_coordinates):
        visited = set()
        queue = deque([(0, agent_coord, 0)])

        obstacle_coordinates_set = set(obstacle_coordinates)

        while queue:
            _, current_coord, g_value = queue.popleft()

            if current_coord == target_coord:
                return True

            if current_coord not in visited:
                visited.add(current_coord)
                neighbors = self.get_neighbors(current_coord, width, height, obstacle_coordinates_set)

                for neighbor in neighbors:
                    if neighbor not in visited:
                        new_g_value = g_value + 1
                        f_value = new_g_value + self.heuristic(neighbor, target_coord)
                        queue.append((f_value, neighbor, new_g_value))
                        queue = deque(sorted(queue, key=lambda x: x[0]))

        return False

    def init_param(self):
        self.MC_iter = 0
        self.run_time = 500  # Run run_time steps per game
        self.map_size = 50
        self.drone_num = 2
        # The area explored by each agent each step
        self.average_list = [0] * self.drone_num
        self.last_drone_pos = []
        self.obstacle_gain_per_agent = np.zeros(self.drone_num)
        self.last_obstacle_gain_per_agent = np.zeros(self.drone_num)
        self.find_grid_count = np.zeros(self.drone_num)
        self.last_find_grid_cout = np.zeros(self.drone_num)
        self.tree_num = 3
        self.human_init_pos = []
        self.human_num = 5
        self.human_num_temp = self.human_num
        self.human_num_copy = self.human_num
        self.sensing_threshold = [3, 5]
        self.time_stamp = None
        self.observed_zone = {}  # 带有时序的已观测点
        self.global_reward = []
        self.global_done = []
        self.per_observed_goal_num = None
        self.obstacles = []  # 记录所有机器人观测到的障碍物
        # 障碍物在地图中的面积占比
        self.obstacle_percentage = 0.2
        self.obstacles_temp = []
        self.human_act_list = []
        self.drone_act_list = []
        self.joint_map = np.zeros((3, self.map_size, self.map_size))
        # initialize trees
        self.land_mark_map = np.zeros((self.map_size, self.map_size))  # 地标地图
        self.memory_step = 1
        self.global_obs_num = 0
        self.t_u = 50
        self.move_threshold = 2
        self.random_pos_robot = True
        self.random_pos_target = True
        self.last_n_step_pos = None
        self.n_step = 4
        view_range_2 = (self.view_range + 2) ** 2
        self.collision = np.zeros(self.drone_num) # 记录which drone take an action that will cause collision
        # intialize tree(随机生成块状障碍物)
        # for i in range(self.tree_num):
        #     tree_pos = []
        #     temp_pos = [random.randint(0, self.map_size-1), random.randint(0, self.map_size-1)]
        #     while self.land_mark_map[temp_pos[0], temp_pos[1]] != 0:
        #         temp_pos = [random.randint(0, self.map_size-1), random.randint(0, self.map_size-1)]
        #     x, y = temp_pos
        #     if 1 < x < self.map_size - 2 and 1 < y < self.map_size - 2:
        #         tree_pos = [[x-1,y-1], [x-1,y], [x-1,y+1], [x,y-1], [x,y],\
        #         [x,y+1], [x+1,y-1], [x+1,y], [x+1,y+1]]
        #         # self.land_mark_map[temp_pos[0], temp_pos[1]] = 2  # tree
        #     for tree in tree_pos:
        #         self.land_mark_map[tree[0], tree[1]] = 2

        # 固定形状障碍物
        # inverse_wall = [[16, 14], [16, 17], [13, 17], [10, 17], [10, 14],
        #                 [34, 36], [34, 33], [37, 33], [40, 33], [40, 36],
        #                 [16, 36], [16, 33], [13, 33], [10, 33], [10, 36],
        #                 [34, 14], [34, 17], [37, 17], [40, 17], [40, 14]]
        #
        # four_long_wall = [[14, 14], [14, 17], [36, 36], [36, 33], [14, 36], [14, 33],
        #                   [36, 14], [36, 17]]
        layout = Layout(map_size=self.map_size, layout='moderate_1_train')
        # wall = layout.wall()

        wall, free_zones = layout.generate_obstacles_and_free_spaces(height=self.map_size, width=self.map_size, obstacle_percentage=self.obstacle_percentage)
        # for pos in inverse_wall:
        for pos in wall:
            tree_pos = []
            x, y = pos
            self.land_mark_map[x, y] = 2
            # self.joint_map[2, x, y] = 1
        # 添加墙体
        # 设置 land_mark_map 的边界为 2
        self.land_mark_map[0, :] = self.land_mark_map[-1, :] = 2
        self.land_mark_map[:, 0] = self.land_mark_map[:, -1] = 2

        # 设置 joint_map 的边界为 1


        # 计算全局观察数
        self.global_obs_num = np.sum(self.land_mark_map == 2)

        # 初始化无人机
        if self.random_pos_robot:
            self.drone_list = []
            id = 0
            temp_pos, selected_positions = self.pick_random_positions(free_zones, self.drone_num - 1)
            self.drone_list.append(Drones(temp_pos, self.view_range, id, self.map_size))
            self.drone_list.extend(
                [Drones(position, self.view_range, id, self.map_size) for id, position in
                 enumerate(selected_positions, start=1)]
            )
        else:
            temp_pos = [[35, 5], [46, 10], [25, 23], [23, 25], [27, 25]]
            self.drone_list = [
                Drones(pos, self.view_range, i, self.map_size) for i, pos in enumerate(temp_pos)
            ]

        # randomly initialize humans
        if self.random_pos_target:
            self.human_list = []

            for i in range(self.human_num):
                temp_pos = random.choice(free_zones)
                flag_in_agent_range = False  # 判断目标的是否一开始初始化在了智能体初始就能检测到的范围内

                for i_agent in range(self.drone_num):
                    try:
                        if (temp_pos[0] - self.drone_list[i_agent].pos[0]) ** 2 + (
                                temp_pos[1] - self.drone_list[i_agent].pos[1]) ** 2 <= view_range_2:
                            flag_in_agent_range = True
                            break
                    except:
                       print("self.drone_num",self.drone_num)
                       print("len temp_pos",len(temp_pos))


                # avaliable = self.is_path_available(agent_coord=self.drone_list[0].pos, target_coord=temp_pos,
                #                                    width=self.map_size, height=self.map_size,
                #                                    obstacle_coordinates=wall)

                # while self.land_mark_map[temp_pos[0], temp_pos[1]] != 0 or flag_in_agent_range or not avaliable:

                # 计算所有智能体的 view_range_2
                view_range_2_list = [agent.view_range ** 2 for agent in self.drone_list]

                # 使用列表解析构建一个满足条件的点的列表
                filtered_free_zones = [
                    pos for pos in free_zones
                    if not any(
                        (pos[0] - agent.pos[0]) ** 2 + (pos[1] - agent.pos[1]) ** 2 <= view_range_2
                        for agent, view_range_2 in zip(self.drone_list, view_range_2_list)
                    )
                ]

                # 在循环中使用 filtered_free_zones 而不是 free_zones
                for i in range(self.human_num):
                    temp_pos = random.choice(filtered_free_zones)

                # while flag_in_agent_range:
                #
                #     temp_pos = random.choice(free_zones)
                #     # print("temp_pos is", temp_pos, "len free_zones", len(free_zones))
                #     # avaliable = self.is_path_available(agent_coord=self.drone_list[0].pos, target_coord=temp_pos,
                #     #                                    width=self.map_size, height=self.map_size,
                #     #                                    obstacle_coordinates=wall)
                #     # print("循环", avaliable)
                #     flag_in_agent_range = False
                #     for i_agent in range(self.drone_num):
                #         if (temp_pos[0] - self.drone_list[i_agent].pos[0]) ** 2 + (
                #                 temp_pos[1] - self.drone_list[i_agent].pos[1]) ** 2 <= view_range_2:
                #             flag_in_agent_range = True
                #             break
                #
                #     # Remove the selected position from free_zones to avoid repetition
                #     print("or flag_in_agent_range:", flag_in_agent_range)
                #     free_zones.remove(temp_pos)

                self.human_init_pos.append(np.array(temp_pos).copy())
                temp_human = Human(np.array(temp_pos))
                self.human_list.append(temp_human)
        # fixedly initialize humans
        else:
            self.human_list = []
            # temp_pos = [[16, 14], [34, 36], [16, 46], [40, 37], [48, 3]]
            temp_pos = [[45, 13], [38, 38], [16, 46], [35, 10], [48, 3]]
            for i in range(self.human_num):
                temp_human = Human(temp_pos[i])
                self.human_init_pos.append(temp_pos[i].copy())
                self.human_list.append(temp_human)

