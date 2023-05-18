import torch
# from env_Drones import EnvDrones
import random
import sys
import os

sys.path.append(r"d:/code/TRPO-in-MARL")
sys.path.append(r"/home/cx")

from configs.config import get_config
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
import time
import numpy as np

np.set_printoptions(threshold=np.inf)
# from ddpg_cnn import Actor_CNN
import envs.EnvDrone.classic_control.env_Drones2 as search_grid
import gym
from algorithms.actor_critic import Actor as actor

import sys
import torch.nn.functional as F
import rescue
import pickle
# maddpg = torch.load("/home/lmy/Downloads/episode_9900.pth")
num_episodes = 50000  # 一共运行 200 个 episode
record_fre = 10000  # 100 个 episode 记录一次
# env = gym.make("SearchGrid-v0")
# train_path = 'D:\code\\rl_local\RL-in-multi-robot-foraging\marl\light_mappo\envs\\resize_scale_120\\train_data.pickle'
# test_path = 'D:\code\\rl_local\RL-in-multi-robot-foraging\marl\light_mappo\envs\\resize_scale_120\\test_data.pickle'
train_path = os.path.join('/home/cx', 'light_mappo/envs', 'resize_scale_120', 'train_data.pickle')
test_path = os.path.join('D:', '\code', 'resize_scale_120', 'test_data.pickle')
# test_path = os.path.join('D:', '\code', 'resize_scale_120', 'test_data.pickle')
with open(train_path, 'rb') as tp:
    data = pickle.load(tp)

map_num = len(data)
env = search_grid.SearchGrid(map_set=data, map_num=map_num)

path0 = r"/home/cx/mappo_model/happo_57_17/actor_agent0.pt"
path1 = r"/home/cx/mappo_model/happo_57_17/actor_agent1.pt"
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
env.seed(all_args.seed + 0 * 1000)
torch.manual_seed(all_args.seed)
torch.cuda.manual_seed_all(all_args.seed)
np.random.seed(all_args.seed)
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
reputation_threshold = 10000



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

obs, _ = env.reset()
fig = plt.figure()
gs = GridSpec(1, 2, figure=fig)
ax1 = fig.add_subplot(gs[0:1, 0:1])
ax2 = fig.add_subplot(gs[0:1, 1:2])

ax1.imshow(env.get_full_obs())
ax2.imshow(env.get_joint_obs(env.MC_iter))
device = torch.device("cpu")

single_map, _, _, _ = env.state_action_reward_done()
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
    # 假设你的模型是actor_1，LayerNorm层名为feature_norm
    # layer_norm_weight = actor_1.base.feature_norm.weight.detach().numpy()
    # layer_norm_bias = actor_1.base.feature_norm.bias.detach().numpy()
    #
    # if prev_layer_norm_weight is not None and prev_layer_norm_bias is not None:
    #     weight_changed = not np.allclose(prev_layer_norm_weight, layer_norm_weight)
    #     bias_changed = not np.allclose(prev_layer_norm_bias, layer_norm_bias)
    #
    #     if weight_changed or bias_changed:
    #         print("LayerNorm parameters changed!")
    #     else:
    #         print("LayerNorm parameters not changed.")
    #
    # prev_layer_norm_weight = layer_norm_weight
    # prev_layer_norm_bias = layer_norm_bias

    print(env.MC_iter)
    obs = np.array(obs)
    # print("obs.shape",obs.shape)
    # print("maks shape is", mask.shape)
    for i, agent in enumerate(agents):
        action[i], _, _,_= agent(np.expand_dims(obs[i], axis=0), rnn_state, mask[i])  # 每个智能体采用自己的状态做动作
    print("pre - action is", action)
    # action = torch.squeeze(action)
    action = action.astype(int)
    # one_hot_action = np.eye(num_class)[action]
    one_hot_action = action.reshape(-1,1)
    print("one-hot", one_hot_action)
    # print("obs grid each agent", env.find_grid_count)

    # Rescue
    for i in range(env.drone_num):
        grid_agents[i] = env.average_list_true[i]
    # 分别记录两个agent没有探索出新地方的次数
    print("grid agents",grid_agents)
    for i in range(env.drone_num):
        # if last_grid_agents[i] <= grid_agents[i] or grid_agents[i] <=0:
        if grid_agents[i] <= 0:
            agent_repetition[i] = agent_repetition[i] + 1
        else:
            agent_repetition[i] = 0
    last_grid_agents = grid_agents.copy()

    # 当 存在agent连续 5 次都没有探索到新地方的时候，触发rescue机制
    # 将这个 agent 当前所在位置作为 rescue 的起始位置，之后使用 astar 规划到最近的 frontier 去
    for i, repetition in enumerate(agent_repetition):
        if repetition > reputation_threshold:
            print("agent_id is", i)
            # 下面这一行就是rescue algorithm
            # resuce_action_list[i].actions, x, y = generate_path(env=env, id=i)
            resuce_action_list[i].id = i
            agent_repetition[i] = -10000

    # generate_path(env=env,agent_repetition=agent_repetition)

    # Clear the images

    for i in range(env.drone_num):
        if len(resuce_action_list[i].actions)>0:
            # 取首个元素，然后删除，直到所有的元素全部删除，此时即到达A star 的目的地
            # 多agents的rescue 用这个
            # one_hot_action[i] = resuce_action_list[i].actions.pop(0)
            # 单 agent的 rescue 用这个
            one_hot_action = resuce_action_list[i].actions.pop(0)
            resuce_flag = True
            if len(resuce_action_list[i].actions) == 0:
                agent_repetition[i] = 0
            print("okok")
        else:
            resuce_flag = False

    obs, reward, done, info, _, _ = env.step(one_hot_action)
    print("reward is",reward)
    if done[0] is True:
        ax1 = fig.add_subplot(gs[0:1, 0:1])
        ax2 = fig.add_subplot(gs[0:1, 1:2])

    ax1.cla()
    ax2.cla()
    if resuce_flag:
        ax1.plot(y, x, c='r', lw=2)
    ax1.imshow(env.get_full_obs())
    ax2.imshow(env.get_joint_obs(env.MC_iter))
    # print("next_single_map", single_map[1])
    plt.pause(.2)
    plt.draw()
