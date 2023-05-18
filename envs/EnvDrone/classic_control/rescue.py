import numpy as np
def generate_path(env, agent_repetition: np.array, reputation_threshold: int):
    for i, repetition in enumerate(agent_repetition):
        if repetition > reputation_threshold:
            print("agent_repetition", agent_repetition)
            start_pos = env.drone_list[i].pos
            # 开始规划路径，第一步找到这个agent的地图里所有的边界点，为此，我们需要一份同时包含障碍物和已探索区域的地图
            # 此时，map中 value>=2 的点就是障碍物，0<value<=1的点就是已经搜索的区域
            map = env.drone_list[i].whole_map[1] + 2 * env.drone_list[i].whole_map[3]
            # 接下来，我需要找到所有的边界点，其特点为，周围的八个临界点里至少有两个的 value 为 0
            frontier_point = find_nearest_frontier(map, start_pos)
            ax1.scatter(frontier_point[1], frontier_point[0], s=1, c='r')
            # plt.show()
            # time.sleep(10)
            path, action_list = rescue_path(map=env.drone_list[i].whole_map[1],
                                            obstacle_map=env.drone_list[i].whole_map[3], target_x=frontier_point[0] \
                                            , target_y=frontier_point[1], start_x=start_pos[0], start_y=start_pos[1])
            # 画路径
            # 这里的 zip(*path) 将 path 列表中的 (x, y) 元组拆分为两个列表，一个是所有的 x 坐标，一个是所有的 y 坐标
            # 然后 ax1.plot(y, x) 会在 ax1 上画出这条路径。
            # print("path is", path)
            x, y = zip(*path)
            ax1.plot(y, x, c='r', lw=2)
            # print("torch.tensor(action)",torch.tensor(action, dtype=torch.int).long())

            one_hot_action = np.zeros((env.drone_num, num_class))

            for i_action in action_list:
                one_hot_action[i] = i_action
                obs, reward, done, info, _ = env.step(one_hot_action)
                ax1.imshow(env.get_full_obs())
                ax2.imshow(env.get_joint_obs(env.MC_iter))
                plt.pause(.5)
                plt.draw()
            # time.sleep(1000)
            agent_repetition[i] = 0