# -*- coding: utf-8 -*-

import torch
# from env_Drones import EnvDrones
import random
import sys
import os

sys.path.append(r"d:/code/TRPO-in-MARL")
sys.path.append(r"/home/cx")
sys.path.append(r"/home/ubuntu/autodl_one_layer")

from configs.config import get_config
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec

import time
import numpy as np

np.set_printoptions(threshold=np.inf)
# from ddpg_cnn import Actor_CNN
import envs.EnvDrone.classic_control.env_Drones_available as search_grid
import gym
from algorithms.actor_critic import Actor as actor

import sys
import torch.nn.functional as F
import rescue
import pickle
# maddpg = torch.load("/home/lmy/Downloads/episode_9900.pth")

# 在程序的开头清空数据记录
happo_cu_data_dir = "/home/ubuntu/autodl_one_layer/envs/EnvDrone/classic_control/happo_cu_data.txt"
frontier_dir = "/home/ubuntu/autodl_one_layer/envs/EnvDrone/classic_control/frontier.txt"
happo_cu_data_dir_dynamic = "/home/ubuntu/autodl_one_layer/envs/EnvDrone/classic_control/happo_cu_data_dynamic.txt"
frontier_dir_dynamic = "/home/ubuntu/autodl_one_layer/envs/EnvDrone/classic_control/frontier_dynamic.txt"
frontier_dir_dynamic_2 = "/home/ubuntu/autodl_one_layer/envs/EnvDrone/classic_control/frontier_dynamic_2.txt"
happo_no_cu_collision = "/home/ubuntu/autodl_one_layer/envs/EnvDrone/classic_control/happo_no_cu_collision.txt"
happo_no_cu_no_collision_200 = "/home/ubuntu/autodl_one_layer/envs/EnvDrone/classic_control/happo_no_cu_no_collision_200.txt"
with open(happo_no_cu_collision, "w") as f:
    pass

num_episodes = 50000  # 一共运行 200 个 episode
record_fre = 10000  # 100 个 episode 记录一次
# env = gym.make("SearchGrid-v0")
# train_path = 'D:\code\\rl_local\RL-in-multi-robot-foraging\marl\light_mappo\envs\\resize_scale_120\\train_data.pickle'
# test_path = 'D:\code\\rl_local\RL-in-multi-robot-foraging\marl\light_mappo\envs\\resize_scale_120\\test_data.pickle'
train_path = os.path.join('/home/ubuntu/autodl_one_layer', 'light_mappo/envs', 'resize_scale_120', 'train_data.pickle')

test_path = os.path.join('/home/ubuntu/autodl_one_layer/envs/resize_scale_120/', 'test_data.pickle')
# test_path = os.path.join('D:', '\code', 'resize_scale_120', 'test_data.pickle')
with open(train_path, 'rb') as tp:
    data = pickle.load(tp)

map_num = len(data)
env = search_grid.SearchGrid(map_set=data, map_num=map_num)

# path0 = r"/home/ubuntu/autodl_one_layer/mappo_model/happo_57_17/actor_agent0.pt"
# path1 = r"/home/ubuntu/autodl_one_layer/mappo_model/happo_57_17/actor_agent1.pt"
path0 = r"/home/ubuntu/autodl_one_layer/mappo_model/happo_57_1/actor_agent_copy0.pt"
path1 = r"/home/ubuntu/autodl_one_layer/mappo_model/happo_57_1/actor_agent_copy1.pt"
prev_layer_norm_weight = None
prev_layer_norm_bias = None

checkpoint0 = torch.load(path0, map_location=torch.device('cpu'))
checkpoint1 = torch.load(path1, map_location=torch.device('cpu'))
# checkpoint = torch.load('D:/code/rl_local/RL-in-multi-robot-foraging/marl/mappo_model/cnn_ICMrun_02/actor.pt', map_location=torch.device('cpu'))
# checkpoint = torch.load('/home/lmy/ArcLab/Projects/multi-robot_foraging/MADDPG/models/old_model/episode_2999.pth')
# actor_1 = light_mappo.algorithms.algorithm.r_actor_critic().eval()

def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str, default='MyEnv', help="Which scenario to run on")
    parser.add_argument("--num_landmarks", type=int, default=4)
    parser.add_argument('--num_agents', type=int, default=2, help="number of players")

    all_args = parser.parse_known_args(args)[0]

    return all_args

parser = get_config()

all_args = parse_args(sys.argv[1:], parser)
env.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
np.random.seed(1)
actor_1 = actor(all_args, env.observation_space, env.action_space)
#
# print("Actor state_dict keys:")
# for key in actor_1.state_dict().keys():
#     print(key)
#
# print("\nCheckpoint state_dict keys:")
# for key in checkpoint0.keys():
#     print(key)

actor_1.load_state_dict(checkpoint0)
actor_1 = actor_1.eval()
actor_2 = actor(all_args, env.observation_space, env.action_space)

actor_2.load_state_dict(checkpoint1)
actor_2 = actor_2.eval()

agents = [actor_1, actor_2]
# agents = [actor_2, actor_1]
# indices = np.arange(env.num_agents)
# acotr = [actor[i] for i in indices]
reputation_threshold = 100



class rescue_action():
    def __init__(self, actions, id):
        self.actions = actions
        self.id = id

# This function is for find the frontier points when an agent continues failing in exploring new areas.
def find_frontier(current_x, current_y, map):
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


def rescue_path(map, obstacle_map, target_x, target_y, start_x, start_y):
    astar = rescue.AStar(obstacle_map=obstacle_map, map=map, target_x=target_x, target_y=target_y, start_x=start_x,
                         start_y=start_y)
    return astar.RunAndSaveImage()


from queue import PriorityQueue


def find_nearest_frontier(map, start):
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

def generate_path(env, id):
    print("agent_repetition", agent_repetition)
    start_pos = env.drone_list[id].pos
    # 开始规划路径，第一步找到这个agent的地图里所有的边界点，为此，我们需要一份同时包含障碍物和已探索区域的地图
    # 此时，map中 value>=2 的点就是障碍物，0<value<=1的点就是已经搜索的区域
    map = env.drone_list[id].whole_map[1] + 2 * env.drone_list[id].whole_map[3]
    # 接下来，我需要找到所有的边界点，其特点为，周围的八个临界点里至少有两个的 value 为 0
    frontier_point = find_nearest_frontier(map, start_pos)
    ax1.scatter(frontier_point[1], frontier_point[0], s=1, c='r')
    # plt.show()
    # time.sleep(10)
    path, action_list = rescue_path(map=env.drone_list[id].whole_map[1],
                                    obstacle_map=env.drone_list[id].whole_map[3], target_x=frontier_point[0] \
                                    , target_y=frontier_point[1], start_x=start_pos[0], start_y=start_pos[1])
    # 画路径
    # 这里的 zip(*path) 将 path 列表中的 (x, y) 元组拆分为两个列表，一个是所有的 x 坐标，一个是所有的 y 坐标
    # 然后 ax1.plot(y, x) 会在 ax1 上画出这条路径。
    # print("path is", path)

    x, y = zip(*path)
    # ax1.plot(y, x, c='r', lw=2)
    return action_list, x, y


def get_action(obs, robot_pos):   #将状态输入模型得到动作
    available_actions = np.ones((2,4))
    obs = obs.reshape(2,60,60)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上、下、左、右
    # self.get_logger().info(f'robot_pos[0]: {robot_pos[0]}')
    for i in range(2):  # 对于两个机器人
        for j, (dx, dy) in enumerate(directions):  # 对于每个方向
            if obs[i][robot_pos[i][0] + dx, robot_pos[i][1] + dy] in [1, 5.5, -4.5]:
                available_actions[i, j] = 0
    return available_actions
                
obs, _ = env.reset()
ax3_image = obs
fig = plt.figure()
gs = GridSpec(2, 2, figure=fig)
ax1 = fig.add_subplot(gs[0:1, 0:1])
ax2 = fig.add_subplot(gs[0:1, 1:2])
ax3 = fig.add_subplot(gs[1:2, 0:1])
ax4 = fig.add_subplot(gs[1:2, 1:2])

ax1_image = env.get_full_obs()
ax2_image = env.get_joint_obs(env.MC_iter) 
ax3_image = np.ones((env.map_size, env.map_size, 3)) * 0.5
ax4_image = ax1_image
for i, drone in enumerate(env.drone_list):
    ax4_image[drone.pos[0], drone.pos[1]] = [0 , 0.25 *i, 0] 


# 找到 ax2_image 中所有非 0.5 的元素的索引
indices = np.where(ax2_image != 0.5)

# 将 ax2_image 中对应位置的值赋给 ax3_image
ax3_image[indices] = ax2_image[indices]

ax1.imshow(ax1_image)
ax2.imshow(ax2_image)    # ax3.cla()
    # ax4.cla()
ax3.imshow(ax3_image)
ax4.imshow(ax4_image)

device = torch.device("cpu")

single_map, _, _, _, available_actions= env.state_action_reward_done(None)
num_class = 4
mask = np.ones((2,1))
rnn_state = np.zeros((1,4,512))
last_grid_agents = np.zeros(env.drone_num)
agent_repetition = np.zeros(env.drone_num)

resuce_action_list = []
grid_agents =[]
for i in range(env.drone_num):
    # id starts from 0.
    resuce_action_list.append(rescue_action(actions=[], id=i))
    grid_agents.append(0)

action = np.empty(2)


                
for i_episode in range(1, num_episodes):

    # print("环境步",env.MC_iter)
    obs = np.array(obs)
    # print("obs shape",obs.shape)
    robot_pos = [env.drone_list[i].pos for i in range(env.drone_num)] # 两个机器人的位置
    # available_actions = get_action(obs, robot_pos)
    print(available_actions)
    for i, agent in enumerate(agents):
        action[i], _, _,_= agent(np.expand_dims(obs[i], axis=0), rnn_state, mask[i], available_actions =  np.expand_dims(available_actions[i], axis=0))  # 每个智能体采用自己的状态做动作
    # print("pre - action is", action)
    action = action.astype(int)
    one_hot_action = action.reshape(-1,1)
    obs, reward, done, info, _, _, ax2_image, available_actions= env.step(one_hot_action)
    # print("reward is",reward)
    ax1_image = env.get_full_obs()
    
    
    # ax2_image = env.get_joint_obs(env.MC_iter)
    # ax5_image = [np.ones((60, 60, 3)) * 0.5 for i in range(env.drone_num)]
    # for i, obs_i in enumerate(obs):
    #     ax5_image[i][np.where(obs_i)] = 0
    
    
    for i in range(env.drone_num):
        # print(len(env.rescue_action_list[i].actions), "is len")
        if len(env.rescue_action_list[i].actions) > 0:
            ax1_image[env.goal_r[i], env.goal_c[i]] = [1, 0, 1]
            # print(env.drone_list[i].path, "is path")
            for j in range(len(env.rescue_action_list[i].actions)): 
                ax1_image[env.drone_list[i].path[j][0], env.drone_list[i].path[j][1]] = [0.7, 0.1, 0.5]
                
            ax1_image[env.drone_list[i].pos[0], env.drone_list[i].pos[1]] =  [0.5 * i, 0, 0.5 * i]
            
    if done[0] is True:
        ax1 = fig.add_subplot(gs[0:1, 0:1])
        ax2 = fig.add_subplot(gs[0:1, 1:2])
        ax3 = fig.add_subplot(gs[1:2, 0:1])
        ax4 = fig.add_subplot(gs[1:2, 1:2])
        ax3.cla()
        ax4.cla()
        ax2_image = np.full((env.map_size, env.map_size, 3), 0.5)
        # 重新初始化 ax3_image 和 ax4_image
        ax3_image = np.ones((env.map_size, env.map_size, 3)) * 0.5
        ax4_image = ax1_image
        for i, drone in enumerate(env.drone_list):
            ax4_image[drone.pos[0], drone.pos[1]] = [0 , 0.25 *i, 0] 

        # print(done, ' is done!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11')

    ax1.cla()
    ax2.cla()
    ax3.cla()
    ax4.cla()

    
    # 找到 ax2_image 中所有非 0.5 的元素的索引
    indices = np.where(ax2_image != 0.5)
    # 将 ax2_image 中对应位置的值赋给 ax3_image
    ax3_image[indices] = ax2_image[indices]
    for i in range(env.drone_num):
        # print(len(env.rescue_action_list[i].actions), "is len")
        if len(env.rescue_action_list[i].actions) > 0:
            ax3_image[env.goal_r[i], env.goal_c[i]] = [1, 0, 1]
            print(env.drone_list[i].path, "is path")
            for j in range(len(env.rescue_action_list[i].actions)):
                ax3_image[env.drone_list[i].path[j][0], env.drone_list[i].path[j][1]] = [0.7, 0.1, 0.5]
                
            ax3_image[env.drone_list[i].pos[0], env.drone_list[i].pos[1]] =  [0.5 * i, 0, 0.5 * i]
    for i, drone in enumerate(env.drone_list):
        if i ==0:
            ax4_image[drone.pos[0], drone.pos[1]] = [0 , 1, 0]
        elif i ==1:
            ax4_image[drone.pos[0], drone.pos[1]] = [0 , 0, 1]
      
    # 画图
      
    ax1.imshow(ax1_image)
    ax2.imshow(ax2_image)
    ax3.imshow(ax3_image)
    ax3.axis('off')
    ax4.imshow(ax4_image)
    plt.pause(.2)
    plt.draw()
    # while True:
    #     user_input = input("请按 '空格' 或 'a' 键继续...")
    #     if user_input in [' ', 'a']:
    #         break




