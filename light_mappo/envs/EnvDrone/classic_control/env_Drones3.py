# 在这个环境里, 智能体一开始不会面临一条路是cutting road 的局面,也就是说, 一开始的几条路都可以遍历全图
# 不会出现一开始的必经之路就只有一条

from typing import Optional
import gym
from gym import spaces
from gym.envs.registration import register
from gym.utils import seeding
import numpy as np
import random
import copy


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
        self.whole_map = np.zeros((4, map_size, map_size), dtype=np.float32)  # 每个机器人保存一个本地地图
        self.last_whole_map = None
        self.grid_communication = 0
        self.obstacle_communication = 0


class Human(object):
    def __init__(self, pos):
        self.pos = pos


class Layout(object):
    def __init__(self, map_size, layout):
        self.map_size = map_size
        self.layout = layout

    def wall(self):
        if self.map_size == 50:
            if self.layout == 'four long wall':
                walls = [[14, 14], [14, 17], [36, 36], [36, 33], [14, 36], [14, 33],
                         [36, 14], [36, 17]]
            if self.layout == 'inverse wall':
                walls = [[16, 14], [16, 17], [13, 17], [10, 17], [10, 14],
                         [34, 36], [34, 33], [37, 33], [40, 33], [40, 36],
                         [16, 36], [16, 33], [13, 33], [10, 33], [10, 36],
                         [34, 14], [34, 17], [37, 17], [40, 17], [40, 14]]
            if self.layout == 'indoor':
                walls = [[8, 8], [8, 11], [8, 14], [11, 14], [11, 17], [17, 17],
                         [17, 14], [20, 14], [20, 11], [20, 8], [11, 20], [11, 23],
                         [11, 26], [11, 29], [11, 32], [8, 32], [8, 35], [8, 38], [8, 40],
                         [17, 23], [17, 20], [17, 29], [17, 32], [20, 32], [20, 35],
                         [20, 38], [20, 40], [20, 23], [23, 23], [26, 23], [29, 23],
                         [29, 20], [29, 17], [29, 14], [29, 8], [29, 5], [29, 2],
                         [32, 14], [35, 14], [38, 14], [41, 14], [41, 17], [41, 20],
                         [41, 23], [41, 26],  [41, 29], [41, 32],
                         [41, 35], [41, 38], [38, 38], [35, 38], [38, 26], [35, 26],
                         [32, 38], [32, 41], [32, 44], [47, 38]]

            return walls


class SearchGrid(gym.Env):
    def __init__(self):
        # self.observation_space = spaces.Box(low=0, high=1, shape=(4, 50, 50))
        # self.action_space = spaces.Discrete(4)
        # 注意，当我用evaluate的时候，使用的是下面两行，当我运行train的时候，暂时使用的是上面两行，下面两行是否可行，暂时没有测试
        # 补充，经过测试，发现似乎确实可行，那么就暂时决定grid_drone就这么用了
        # When use mlp
        self.observation_space = spaces.Box(low=0, high=1, shape=(4 * 50 * 50,))
        self.share_observation_space = spaces.Box(low=0, high=1, shape=(3 * 50 * 50,))
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
        # reward_mean = np.mean(reward)
        for i in range(len(reward)):
            # reward[i] = reward_mean
            reward[i] = reward[i]*2
        # print("MC_iter",self.MC_iter)
        # if reward != -2:
        #     print(f"reward:{reward}, action:{action}")
        # 当使用mlp时，将observation给flatten，当使用CNN时，不需要
        for i in range(len(observation)):
            observation[i] = observation[i].flatten()
        return observation, reward, done, info, self.joint_map.flatten()

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False,
              options: Optional[dict] = None):
        self.init_param()
        self.MC_iter = self.MC_iter + 1
        # self.get_full_obs()
        self.get_joint_obs(self.MC_iter)
        for i in range(self.drone_num):
            self.last_drone_pos.append(self.drone_list[i].pos)
        observation, _, _, info = self.state_action_reward_done()
        # print("observation",observation[0].shape)
        # print("This is reset!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # 当使用mlp时，使用flatten。当使用CNN时，注释掉
        for i in range(len(observation)):
            observation[i] = observation[i].flatten()

        # 当使用mlp时，才需要将joint_map flatten，否则是不需要的
        # print("!!!!!!!!shape of joint map",self.joint_map.shape)
        return observation, self.joint_map.flatten()
        # return (observation[0])

    def render(self):
        pass

    def close(self):
        pass

    def drone_step(self, drone_act_list):
        # drone_act_list = [drone_act_list]
        # for k in range(self.drone_num):
        #     if drone_act_list[k][0] == 1:
        #         self.drone_list[k].pos[0] = self.drone_list[k].pos[0] - 1
        #         # print("1")
        #     elif drone_act_list[k][1] == 1:
        #         self.drone_list[k].pos[0] = self.drone_list[k].pos[0] + 1
        #         # print("2")
        #     elif drone_act_list[k][2] == 1:
        #         self.drone_list[k].pos[1] = self.drone_list[k].pos[1] - 1
        #         # print("3")
        #     elif drone_act_list[k][3] == 1:
        #         self.drone_list[k].pos[1] = self.drone_list[k].pos[1] + 1
        #         # print("4")
        #     # print(self.drone_list[k].pos[0], self.drone_list[k].pos[1])

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

    def human_step(self, human_act_list):
        for k in range(self.human_num):
            # print(self.human_init_pos)
            # print([self.human_list[k].pos[0]-self.human_init_pos[k][0], self.human_list[k].pos[1]-self.human_init_pos[k][1]])
            if human_act_list[k] == 0:
                if self.human_list[k].pos[0] > 0 and (self.human_list[k].pos[0] - \
                                                      self.human_init_pos[k][0] - 1 > -self.move_threshold):
                    free_space = self.land_mark_map[self.human_list[k].pos[0] - 1, self.human_list[k].pos[1]]
                    if free_space == 0:
                        self.human_list[k].pos[0] = self.human_list[k].pos[0] - 1
            elif human_act_list[k] == 1:
                if self.human_list[k].pos[0] < self.map_size - 1 and (self.human_list[k].pos[0] - \
                                                                      self.human_init_pos[k][
                                                                          0] + 1 < self.move_threshold):
                    free_space = self.land_mark_map[self.human_list[k].pos[0] + 1, self.human_list[k].pos[1]]
                    if free_space == 0:
                        self.human_list[k].pos[0] = self.human_list[k].pos[0] + 1
            elif human_act_list[k] == 2:
                if self.human_list[k].pos[1] > 0 and (self.human_list[k].pos[1] - \
                                                      self.human_init_pos[k][1] - 1 > -self.move_threshold):
                    free_space = self.land_mark_map[self.human_list[k].pos[0], self.human_list[k].pos[1] - 1]
                    if free_space == 0:
                        self.human_list[k].pos[1] = self.human_list[k].pos[1] - 1
            elif human_act_list[k] == 3:
                if self.human_list[k].pos[1] < self.map_size - 1 and (self.human_list[k].pos[1] - \
                                                                      self.human_init_pos[k][
                                                                          1] + 1 < self.move_threshold):
                    free_space = self.land_mark_map[self.human_list[k].pos[0], self.human_list[k].pos[1] + 1]
                    if free_space == 0:
                        self.human_list[k].pos[1] = self.human_list[k].pos[1] + 1

    def get_full_obs(self):  # 这里是整个环境的信息
        obs = np.ones((self.map_size, self.map_size, 3))
        for i in range(self.map_size):
            for j in range(self.map_size):
                if self.land_mark_map[i, j] == 1:  # [0,0,0]表示wall
                    obs[i, j] = 0
                if self.land_mark_map[i, j] == 2:  # [0,1,0]表示tree
                    obs[i, j] = 0

        for i in range(self.human_num):  # [1,0,0]表示human
            obs[self.human_list[i].pos[0], self.human_list[i].pos[1], 0] = 1
            obs[self.human_list[i].pos[0], self.human_list[i].pos[1], 1] = 0
            obs[self.human_list[i].pos[0], self.human_list[i].pos[1], 2] = 0

        for i in range(self.drone_num):
            obs[self.drone_list[i].pos[0], self.drone_list[i].pos[1], 0] = 0.5*i
            obs[self.drone_list[i].pos[0], self.drone_list[i].pos[1], 1] = 0*i
            obs[self.drone_list[i].pos[0], self.drone_list[i].pos[1], 2] = 0.5*i
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

        # drone.whole_map[0, drone.pos[0], drone.pos[1]] = self.memory_step  # 记录100步的信息
        # drone.whole_map[0][drone.whole_map[0]>0] -= 0.1

        # drone.whole_map[0, drone.pos[0], drone.pos[1]] = self.memory_step

        # 先通过通信更新得图：

        # 在观测范围内进行信息更新，更新时间戳地图1，轨迹地图2和障碍物地图3
        # 禁止通信
        for k in range(self.drone_num):
            if self.drone_list[k].id != drone.id and (self.drone_list[k].pos[0] - drone.pos[0]) ** 2 \
                    + (self.drone_list[k].pos[1] - drone.pos[1]) ** 2 <= sensing_size ** 2:
                drone.communicate_list.append(k)
                # 记录其他智能体的位置，以后可以变成记录其他智能体的轨迹
                drone.whole_map[2, self.drone_list[k].pos[0], self.drone_list[k].pos[1]] = 1
                # print("信息交换")
                # print("距离的平方是", (self.drone_list[k].pos[0]-drone.pos[0])**2\
                #     + (self.drone_list[k].pos[1]-drone.pos[1])**2)
                # print("距离阈值是",sensing_size**2)
        drone.grid_communication = 0
        drone.obstacle_communication = 0
        if drone.communicate_list:
            maps = np.array([self.drone_list[i].whole_map for i in drone.communicate_list] + [drone.whole_map])
            max_2 = np.max(maps[:, 2, :, :], axis=0)
            max_1 = np.max(maps[:, 1, :, :], axis=0)
            max_3 = np.max(maps[:, 3, :, :], axis=0)

            update_3 = max_3 > drone.whole_map[3, :, :]
            # update_1 = max_1 > drone.whole_map[1, :, :]

            drone.grid_communication += np.count_nonzero(max_1) - np.count_nonzero(drone.whole_map[1])
            drone.obstacle_communication += np.sum(update_3)
            # print("drone.grid_communication",drone.grid_communication)
            # print("drone.obstacle_communication",drone.obstacle_communication)

            drone.whole_map[2, :, :] = np.maximum(max_2, drone.whole_map[2, :, :])
            drone.whole_map[3, :, :] = max_3
            drone.whole_map[1, :, :] = max_1

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
        for i in range(obs_size):
            for j in range(obs_size):
                x = i + drone.pos[0] - drone.view_range + 1
                y = j + drone.pos[1] - drone.view_range + 1
                if 0 <= x <= self.map_size - 1 and 0 <= y <= self.map_size - 1:
                    if self.land_mark_map[x, y] == 2:
                        obs[i, j] = 0

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
                        if chosen_gap < drone.view_range:
                            if gap[0] >= 0 and gap[1] > 0:
                                if gap[0] == 0:
                                    if obs[i + 1, j, 0] == 0 and obs[i + 1, j, 1] == 0 and \
                                            obs[i + 1, j, 2] == 0:
                                        for num in range(drone.view_range - chosen_gap - 1):
                                            for num_1 in range(num + 2):
                                                drone.unobserved.append([i + num_1, j + num + 1])
                                    if obs[i - 1, j, 0] == 0 and obs[i - 1, j, 1] == 0 and \
                                            obs[i - 1, j, 2] == 0:
                                        for num in range(drone.view_range - chosen_gap - 1):
                                            for num_1 in range(num + 2):
                                                drone.unobserved.append([i - num_1, j + num + 1])
                                else:
                                    for num in range(drone.view_range - chosen_gap - 1):
                                        drone.unobserved.append([i + num + 1, j + num + 1])
                            if gap[0] > 0 and gap[1] <= 0:
                                if gap[1] == 0:
                                    if obs[i, j + 1, 0] == 0 and obs[i, j + 1, 1] == 0 and \
                                            obs[i, j + 1, 2] == 0:
                                        for num in range(drone.view_range - chosen_gap - 1):
                                            for num_1 in range(num + 2):
                                                drone.unobserved.append([i + num + 1, j + num_1])
                                    if obs[i, j - 1, 0] == 0 and obs[i, j - 1, 1] == 0 and \
                                            obs[i, j - 1, 2] == 0:
                                        for num in range(drone.view_range - chosen_gap - 1):
                                            for num_1 in range(num + 2):
                                                drone.unobserved.append([i + num + 1, j - num_1])
                                else:
                                    for num in range(drone.view_range - chosen_gap - 1):
                                        drone.unobserved.append([i + num + 1, j - num - 1])
                            if gap[0] < 0 and gap[1] >= 0:
                                if gap[1] == 0:
                                    if obs[i, j + 1, 0] == 0 and obs[i, j + 1, 1] == 0 and \
                                            obs[i, j + 1, 2] == 0:
                                        for num in range(drone.view_range - chosen_gap - 1):
                                            for num_1 in range(num + 2):
                                                drone.unobserved.append([i - num - 1, j + num_1])
                                    if obs[i, j - 1, 0] == 0 and obs[i, j - 1, 1] == 0 and \
                                            obs[i, j - 1, 2] == 0:
                                        for num in range(drone.view_range - chosen_gap - 1):
                                            for num_1 in range(num + 2):
                                                drone.unobserved.append([i - num - 1, j - num_1])
                                else:
                                    for num in range(drone.view_range - chosen_gap - 1):
                                        drone.unobserved.append([i - num - 1, j + num + 1])
                            if gap[0] <= 0 and gap[1] < 0:
                                if gap[0] == 0:
                                    if obs[i + 1, j, 0] == 0 and obs[i + 1, j, 1] == 0 and \
                                            obs[i + 1, j, 2] == 0:
                                        for num in range(drone.view_range - chosen_gap - 1):
                                            for num_1 in range(num + 2):
                                                drone.unobserved.append([i + num_1, j - num - 1])
                                    if obs[i - 1, j, 0] == 0 and obs[i - 1, j, 1] == 0 and \
                                            obs[i - 1, j, 2] == 0:
                                        for num in range(drone.view_range - chosen_gap - 1):
                                            for num_1 in range(num + 2):
                                                drone.unobserved.append([i - num_1, j - num - 1])
                                else:
                                    for num in range(drone.view_range - chosen_gap - 1):
                                        drone.unobserved.append([i - num - 1, j - num - 1])
                    else:
                        # 对观测到的区域添加时间戳
                        drone.whole_map[1, x, y] = 1
                else:  # 其他情况
                    obs[i, j] = 0.5

                # 这里是设置圆形观测区域
                if (drone.view_range - 1 - i) * (drone.view_range - 1 - i) + (drone.view_range - 1 - j) * (
                        drone.view_range - 1 - j) > drone.view_range * drone.view_range:
                    obs[i, j] = 0.5

        for pos in drone.unobserved:  # 这里处理后得到的obs是能观测到的标志物地图
            obs[pos[0], pos[1]] = 0.5

        # 这里计算与其他机器人在时刻t的相对位置
        # temp_list = []
        # for i in range(self.drone_num):
        #     temp = ((self.drone_list[i].pos[0] - drone.pos[0]) ** 2 + \
        #             (self.drone_list[i].pos[1] - drone.pos[1]) ** 2) ** 0.5
        #     if i != drone.id:
        #         temp_list.append(temp)
        # drone.relative_pos = temp_list
        # # print("relative_pos:",drone.relative_pos)
        # # 这里计算与其他机器人在时刻t的相对方向
        # temp_list = []
        # for i in range(self.drone_num):
        #     temp = self.get_relative_direction(drone.pos[0], drone.pos[1], \
        #                                        self.drone_list[i].pos[0], self.drone_list[i].pos[1])
        #     if i != drone.id:
        #         temp_list.append(temp)
        # drone.relative_direction = temp_list

        # drone.whole_map[0] = np.zeros(drone.whole_map[0].shape, dtype=np.float32)
        # drone.whole_map[0, drone.pos[0], drone.pos[1]] = 1
        # for i in range(self.map_size):  # 这里进行轨迹的衰减
        #     for j in range(self.map_size):
        #         if drone.whole_map[2, i, j] > 1 / self.t_u:
        #             drone.whole_map[2, i, j] = drone.whole_map[2, i, j] - 1 / self.t_u
        #         # 进行自身探索地图的衰减
        #         drone.whole_map[1, i, j] = max(0, drone.whole_map[1, i, j]-0.01)

        # 进行轨迹的衰减, 是上面那段注释的优化
        drone.whole_map[2, :, :] -= 1 / self.t_u
        drone.whole_map[2, drone.whole_map[2, :, :] < 1 / self.t_u] = 0
        # 进行自身探索地图的衰减
        drone.whole_map[1, :, :] = np.maximum(0, drone.whole_map[1, :, :] - 0.01)

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
        obs = np.ones((self.map_size, self.map_size, 3))
        for i in range(self.map_size):
            for j in range(self.map_size):
                obs[i, j] = 0.5
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

        # print("处理前：", self.obstacles)
        # for pos in self.obstacles:   #这种方法可能指针会跳跃，导致输出的结果不一定正确
        #     while self.obstacles.count(pos) > 1:
        #         self.obstacles.remove(pos)
        # 原本是用来去掉重复观测的障碍物的，但是现在去重复这一步在前面就做了，重复的没有纳入计数，所以这里就不用了
        # temp_list_copy = copy.deepcopy(self.obstacles)
        # for pos in temp_list_copy:
        #     while self.obstacles.count(pos) > 1:
        #         self.obstacles.remove(pos)

        # 暂时也不用相对距离了
        # for k in range(self.drone_num):  # 这里计算与观测区域内所有机器人的相对距离
        #     temp_list = []
        #     # temp = [0, 0]
        #     for pos in self.drone_list[k].observed_drone:
        #         temp_list.append([abs(self.drone_list[k].pos[0] - pos[0]), \
        #                           abs(self.drone_list[k].pos[1] - pos[1])])
            # for pos in temp_list:
            #     temp[0] += pos[0] / len(temp_list)
            #     temp[1] += pos[1] / len(temp_list)
            # self.drone_list[k].relative_coordinate = temp

            # 这是对上面代码的优化:
            # temp_arr = np.array(temp_list)
            # temp = np.mean(temp_arr, axis=0)
            # self.drone_list[k].relative_coordinate = temp
        # sensing_size = 2 * self.drone_list[0].view_range - 1
        # index = random.randint(self.sensing_threshold[0], self.sensing_threshold[1])
        # for k in range(self.drone_num):  # 更新观测范围内的时间戳地图
        #     for i in range(sensing_size):
        #         for j in range(sensing_size):

        #             x = i + self.drone_list[k].pos[0] - (self.drone_list[k].view_range + index) + 1
        #             y = j + self.drone_list[k].pos[1] - (self.drone_list[k].view_range + index) + 1
        #             if 0 <= x < 50 and 0 <= y < 50:
        #                 self.drone_list[k].whole_map[1, x, y] = self.MC_iter
        # 合并所有无人机的整个地图
        for drone in self.drone_list:
            self.joint_map[0, drone.pos[0], drone.pos[1]] = 5
        self.joint_map[0, :, :] = np.maximum(0,  self.joint_map[0, :, :] - 0.01)
        # self.joint_map[0] = np.max([drone.whole_map[0] for drone in self.drone_list], axis=0)
        self.joint_map[1] = np.max([drone.whole_map[1] for drone in self.drone_list], axis=0)

        return obs

    def state_action_reward_done(self):  # 这里返回状态值，奖励值，以及游戏是否结束
        # print("reward is")
        # reward = 0  # 合作任务，只设置单一奖励
        # reward_list = np.zeros(self.drone_num, dtype=np.float32)
        ####################设置奖励的增益
        target_factor = 200
        # 发现障碍物的奖励系数
        information_gain = 0.5
        # distance_factor = 0.05
        # pos_without_change_factor = 20
        # time step factor 变成 0, 取消时间惩罚
        # 时间惩罚
        time_step_factor = 1
        # 发现新区域的奖励系数
        average_time_stamp_factor = 0.2
        collision_factor = 500

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
        # if self.MC_iter > 0:
        #     # 这里对状态进行最值归一化，只需要操作观测历史层
        #     for single_map in single_map_set:
        #         max_value = np.max(single_map[1])
        #         # Question: Why the state normalization cause a fail on calculating actions?
        #         # print("sigle_map [1] is", single_map[1])
        #         single_map[1] = single_map[1] / max_value

        info = {}
        reward_list = [target_factor * i_agent for i_agent in self.target_per_agent]  # 这里计算发现目标点的数量
        # print("reward
        # ,", reward_list)

        if self.human_num == 0:
            reward_list = list(map(lambda x: x + 500, reward_list))
            done = True
            # info['0'] = "find all target"
        for i in range(self.drone_num - 1):  # 如果机器人发生碰撞
            for j in range(i + 1, self.drone_num):
                distance = np.linalg.norm(self.drone_list[i].pos - self.drone_list[j].pos)
                if distance <= 1:
                    done = True
                    reward_list[i] -= collision_factor
                    reward_list[j] -= collision_factor

                    # print("collisioin with robots", -collision_factor)
                    # info['0'] = "robot collision"
        # for i in range(self.drone_num):  # 如果机器人和障碍物发生碰撞
        #     for j in range(len(self.obstacles)):
        #         if self.drone_list[i].pos[0] == self.obstacles[j][0] and \
        #                 self.drone_list[i].pos[1] == self.obstacles[j][1]:
        #             done = True
        #             reward_list[i] -= collision_factor
        # print("collision with obstacles", -200)
        # info['0'] = "obs collision"

        # 这是对上面的优化
        for i, drone in enumerate(self.drone_list):
            if self.land_mark_map[drone.pos[0], drone.pos[1]] > 0:
                # # 如果碰撞, 使无人机位置和上一轮一样,从而保持不动
                done = True
                # print("reward list",reward_list)
                # print("drone_id",drone.id)
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
        # for i in drone.relative_pos:  # 这里计算相对位置的奖励
        #     sum_1 += i
        # sum_1 = pos_factor * sum_1
        # reward += sum_1 / 2 / self.drone_num
        # sum_2 = 0
        # sorted_relative_direction = copy.deepcopy(drone.relative_direction)
        # sorted_relative_direction.sort()
        # # print("sorted: ", sorted_relative_direction)
        # # print("original: ", drone.relative_direction)
        # for i in range(len(sorted_relative_direction) - 1):  # 这里计算相对方向的奖励
        #     sum_2 += sorted_relative_direction[i + 1] - sorted_relative_direction[i]
        # reward += direction_factor * sum_2 / self.drone_num

        # 这里增加通信频次更新信息的奖励
        # sum_4 = drone.communicate_rate / 2 / self.drone_num * communicate_factor
        # reward += sum_4

        return reward

    def init_param(self):
        self.MC_iter = 0
        self.run_time = 2000  # Run run_time steps per game
        self.map_size = 50
        self.drone_num = 2
        # The area explored by each agent each step
        self.average_list = [0] * self.drone_num
        self.last_drone_pos = []
        self.obstacle_gain_per_agent = np.zeros(self.drone_num)
        self.last_obstacle_gain_per_agent = np.zeros(self.drone_num)
        self.find_grid_count = np.zeros(self.drone_num)
        self.last_find_grid_cout = np.zeros(self.drone_num)
        self.view_range = 10
        # self.view_range = 6
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
        layout = Layout(map_size=50, layout='indoor')
        wall = layout.wall()
        # for pos in inverse_wall:
        for pos in wall:
            tree_pos = []
            temp_pos = pos
            x, y = temp_pos
            if 1 < x < self.map_size - 2 and 1 < y < self.map_size - 2:
                tree_pos = [[x - 1, y - 1], [x - 1, y], [x - 1, y + 1], [x, y - 1], [x, y], \
                            [x, y + 1], [x + 1, y - 1], [x + 1, y], [x + 1, y + 1]]
                # self.land_mark_map[temp_pos[0], temp_pos[1]] = 2  # tree
            for tree in tree_pos:
                self.land_mark_map[tree[0], tree[1]] = 2
                # self.
        # 随机生成墙体
        # for i in range(self.tree_num):
        #     tree_pos = []
        #     temp_pos = [random.randint(0, self.map_size - 1), random.randint(0, self.map_size - 1)]
        #     while self.land_mark_map[temp_pos[0], temp_pos[1]] != 0:
        #         temp_pos = [random.randint(0, self.map_size - 1), random.randint(0, self.map_size - 1)]
        #     temp_pos = [[temp_pos[0], temp_pos[1] - 3], [temp_pos[0], temp_pos[1]], [temp_pos[0], temp_pos[1] + 3], \
        #                 [temp_pos[0] + 3, temp_pos[1] - 6], [temp_pos[0] + 6, temp_pos[1] - 3],
        #                 [temp_pos[0] + 6, temp_pos[1]], \
        #                 [temp_pos[0] + 6, temp_pos[1] + 3], [temp_pos[0] + 3, temp_pos[1] + 6], \
        #                 [temp_pos[0], temp_pos[1] - 6], [temp_pos[0], temp_pos[1] + 6], \
        #                 [temp_pos[0] + 6, temp_pos[1] - 6], [temp_pos[0] + 6, temp_pos[1] + 6]]
        #     del temp_pos[random.randint(0, len(temp_pos) - 1)]
        #     del temp_pos[random.randint(0, len(temp_pos) - 1)]
        #     for j in temp_pos:
        #         x = j[0]
        #         y = j[1]
        #         if 1 < x < self.map_size - 2 and 1 < y < self.map_size - 2:
        #             tree_pos = [[x - 1, y - 1], [x - 1, y], [x - 1, y + 1], [x, y - 1], [x, y], \
        #                         [x, y + 1], [x + 1, y - 1], [x + 1, y], [x + 1, y + 1]]
        #             # self.land_mark_map[temp_pos[0], temp_pos[1]] = 2  # tree
        #         for tree in tree_pos:
        #             self.land_mark_map[tree[0], tree[1]] = 2
        # 边缘加上围墙
        for i in range(self.map_size):
            self.land_mark_map[i, 0] = 2
            self.land_mark_map[0, i] = 2
            self.land_mark_map[self.map_size - 1, i] = 2
            self.land_mark_map[i, self.map_size - 1] = 2

            self.joint_map[2, i, 0] = 1
            self.joint_map[2, 0, i] = 1
            self.joint_map[2, self.map_size - 1, i] = 1
            self.joint_map[2, i, self.map_size - 1] = 1

        for i in range(self.map_size):
            for j in range(self.map_size):
                if self.land_mark_map[i, j] == 2:
                    self.global_obs_num += 1
                # randomly initialize drones
                # self.start_pos = [self.map_size-1, self.map_size-1]
            if self.random_pos_robot:
                self.drone_list = []

                # Initiate agents in nearby positions
                # 先随机初始化一个agent 然后在[-drone_num, drone_num]范围内，随机初始化 other agents
                temp_pos = [random.randint(0, self.map_size - 1), random.randint(0, self.map_size - 1)]

                # Check whether the validity of the initiated position
                while self.land_mark_map[temp_pos[0], temp_pos[1]] != 0:
                    temp_pos = [random.randint(0, self.map_size - 1), random.randint(0, self.map_size - 1)]

                # If valid, add the position in Drones
                id = 0
                temp_drone = Drones(temp_pos, self.view_range, id, self.map_size)
                self.drone_list.append(temp_drone)

                # Get the neighbor positions of temp_pos
                options = np.meshgrid(
                    range(max(0, temp_pos[0] - self.drone_num-2), min(self.map_size-1, temp_pos[0] + self.drone_num + 2)),
                    range(max(0, temp_pos[1] - self.drone_num-2), min(self.map_size-1, temp_pos[1] + self.drone_num + 2))
                )

                options = np.stack(options, axis=-1).reshape(-1, 2)
                options = np.delete(options, np.where(np.all(options == temp_pos, axis=1)), axis=0)

                # Get the position of other agents
                for i in range(self.drone_num-1):
                    temp_pos2 = options[np.random.choice(options.shape[0])]

                    # temp_pos2 = [temp_pos[0] + random.randint(0, 2*self.drone_num-1), temp_pos[1] + random.randint(0, 2*self.drone_num-1)]
                    # 如果初始化的位置不是空白区域，或者两个智能体紧紧相邻，则重新生成另一个智能体的位置
                    while self.land_mark_map[temp_pos2[0], temp_pos2[1]] != 0 or np.linalg.norm(temp_pos - temp_pos2)<=1:
                        # temp_pos = [random.randint(0, self.map_size - 1), random.randint(0, self.map_size - 1)]
                        # 删除不符合规定的，然后重新选
                        options = np.delete(options, np.where(np.all(options == temp_pos2, axis=1)), axis=0)
                        temp_pos2 = options[np.random.choice(options.shape[0])]
                    id = id+1
                    temp_drone = Drones(temp_pos2, self.view_range, id, self.map_size)
                    # 删除已经选择过的
                    options = np.delete(options, np.where(np.all(options == temp_pos2, axis=1)), axis=0)
                    self.drone_list.append(temp_drone)
                # fixedly initialize robot
            else:
                self.drone_list = []
                # temp_pos = [[45, 5], [46, 6], [25, 23], [23, 25], [27, 25]]
                temp_pos = [[35, 5], [46, 10], [25, 23], [23, 25], [27, 25]]
                for i in range(self.drone_num):
                    temp_drone = Drones(temp_pos[i], self.view_range, i, self.map_size)
                    self.drone_list.append(temp_drone)

        # randomly initialize humans
        if self.random_pos_target:
            self.human_list = []

            for i in range(self.human_num):
                temp_pos = [random.randint(0, self.map_size - 1), random.randint(0, self.map_size - 1)]
                flag_in_agent_range = False  # 判断目标的是否一开始初始化在了智能体初始就能检测到的范围内
                for i_agent in range(self.drone_num):
                    if (temp_pos[0] - self.drone_list[i_agent].pos[0]) ** 2 + (
                            temp_pos[1] - self.drone_list[i_agent].pos[1]) ** 2 <= view_range_2:
                        flag_in_agent_range = True
                        break
                while self.land_mark_map[temp_pos[0], temp_pos[1]] != 0 or flag_in_agent_range:
                    temp_pos = [random.randint(0, self.map_size - 1), random.randint(0, self.map_size - 1)]
                    flag_in_agent_range = False
                    for i_agent in range(self.drone_num):
                        if (temp_pos[0] - self.drone_list[i_agent].pos[0]) ** 2 + (
                                temp_pos[1] - self.drone_list[i_agent].pos[1]) ** 2 <= view_range_2:
                            flag_in_agent_range = True
                            break

                self.human_init_pos.append(temp_pos.copy())
                temp_human = Human(temp_pos)
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
