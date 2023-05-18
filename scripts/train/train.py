"""
# @Time    : 2021/6/30 10:07 下午
# @Author  : hezhiqiang
# @Email   : tinyzqh@163.com
# @File    : train.py
"""

# !/usr/bin/env python
import sys
import os
import socket
import setproctitle
import numpy as np
import pickle

from pathlib import Path
import torch
import sys
sys.path.append(r"/home/cx")
from config import get_config
from envs.env_wrappers import DummyVecEnv
from envs.env_wrappers_multiprocess import MultiDummyVecEnv
import datetime
import multiprocessing
"""Train script for MPEs."""
# import taichi as ti

def make_train_env(all_args, map_set, map_num):
    def get_env_fn(rank):
        def init_env():
            # TODO 注意注意，这里选择连续还是离散可以选择注释上面两行，或者下面两行。
            # from envs.env_continuous import ContinuousActionEnv
            # env = ContinuousActionEnv()
            from envs.env_discrete import DiscreteActionEnv
            # 假设要读取第一个数组
            env = DiscreteActionEnv(map_set, map_num)
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    # cpu_cors = multiprocessing.cpu_count()
    # print("并行的环境和cpu核的数量一致：", cpu_cors)
    # return DummyVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])
    return MultiDummyVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])

def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            # TODO 注意注意，这里选择连续还是离散可以选择注释上面两行，或者下面两行。
            # from envs.env_continuous import ContinuousActionEnv
            # env = ContinuousActionEnv()
            from envs.env_discrete import DiscreteActionEnv
            env = DiscreteActionEnv()
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    # return DummyVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])
    return MultiDummyVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str, default='MyEnv', help="Which scenario to run on")
    parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument('--num_agents', type=int, default=2, help="number of players")

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):

    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "rmappo":
        print("all_args.use_recurrent_policy",all_args.use_recurrent_policy)
        assert (all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy), ("check recurrent policy!")
    elif all_args.algorithm_name == "mappo":
        print("all_args.use_recurrent_policy",all_args.use_recurrent_policy)
        assert (all_args.use_recurrent_policy == False and all_args.use_naive_recurrent_policy == False), (
            "check recurrent policy!")
    else:
        raise NotImplementedError

    assert (all_args.share_policy == True and all_args.scenario_name == 'simple_speaker_listener') == False, (
        "The simple_speaker_listener scenario can not use shared policy. Please check the config.py.")

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        print("device is",device)
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                       0] + "/results") / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if not run_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = run_dir / curr_run
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + \
                              str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(
        all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    train_path = os.path.join('/home/cx', 'light_mappo/envs', 'resize_scale_120', 'train_data.pickle')
    # test_path = os.path.join('D:', '\code', 'resize_scale_120', 'test_data.pickle')
    with open(train_path, 'rb') as tp:
        data = pickle.load(tp)

    map_num = len(data)


    envs = make_train_env(all_args, map_set=data, map_num=map_num)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }
    
    # run experiments
    if all_args.share_policy:
        from runner.shared.env_runner import EnvRunner as Runner
    else:
        from runner.separated.env_runner import EnvRunner as Runner

    # log_dir_address = '/home/cx/mappo_result/cnn_ICMtest57_02'
    log_dir_address = '/home/cx/mappo_result/cnn_test57_12'
    content_after_last_slash = log_dir_address.split('/')[-1]

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + \
                              str("地图集那个") + "-" + str(all_args.experiment_name) + "@" + str(
        content_after_last_slash))
    if not os.path.exists(log_dir_address):
        print("不存在该路径，正在创建")
        # print(log_dir_address)
        os.makedirs(log_dir_address)
    print(log_dir_address)
    print("Start")
    runner = Runner(config,log_dir_address=log_dir_address)
    runner.run()        

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()
    #
    # runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
    # runner.writter.close()


if __name__ == "__main__":
    # ti.init(arch=ti.gpu)
    start = datetime.datetime.now()
    print("sys argv",sys.argv[1:])
    main(sys.argv[1:])
    end = datetime.datetime.now()
    print("start:", start," end:", end, " total:", end-start)