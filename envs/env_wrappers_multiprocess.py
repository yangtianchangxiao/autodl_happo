import numpy as np
import multiprocessing as mp

# 注意！如果是单智能体，就需要把done[0]全部换成done
class MultiDummyVecEnv():
    def __init__(self, env_fns):
        self.num_envs = len(env_fns)
        self.envs = [fn() for fn in env_fns]
        self.observation_space = self.envs[0].observation_space
        self.share_observation_space = self.envs[0].share_observation_space
        self.action_space = self.envs[0].action_space
        # print("action space is~~~~~~~~~~~~~~",self.action_space)
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(self.num_envs)])
        self.processes = [mp.Process(target=self.worker, args=(work_remote, remote, env)) for work_remote, remote, env in zip(self.work_remotes, self.remotes, self.envs)]

        for process in self.processes:
            process.start()
        for work_remote in self.work_remotes:
            work_remote.close()

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))

    # def step_wait(self):
    #     results = [remote.recv() for remote in self.remotes]
    #     obs, rews, dones, infos, joint_maps = map(np.array, zip(*results))
    #
    #     for (i, done) in enumerate(dones):
    #         if 'bool' in done.__class__.__name__:
    #             if done:
    #                 obs[i], joint_maps[i] = self.envs[i].reset()
    #         else:
    #             if np.all(done):
    #                 obs[i], joint_maps[i] = self.envs[i].reset()
    #
    #     return obs, rews, dones, infos, joint_maps
    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        obs, rews, dones, infos, joint_maps, rescue_masks = map(np.array, zip(*results))

        for (i, done) in enumerate(dones):
            if 'bool' in done.__class__.__name__:
                if done[0]:
                    self.remotes[i].send(('reset', None))
                    obs[i], joint_maps[i] = self.remotes[i].recv()
            else:
                if np.all(done[0]):
                    self.remotes[i].send(('reset', None))
                    obs[i], joint_maps[i] = self.remotes[i].recv()

        return obs, rews, dones, infos, joint_maps, rescue_masks

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        obs, joint_maps = map(np.array, zip(*results))
        return obs, joint_maps

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for process in self.processes:
            process.join()

    # def worker(self, remote, parent_remote, env):
    #     parent_remote.close()
    #     while True:
    #         cmd, data = remote.recv()
    #         if cmd == 'step':
    #             obs, reward, done, info, joint_map = env.step(data)
    #             if done:
    #                 obs, joint_map = env.reset()
    #             remote.send((obs, reward, done, info, joint_map))
    #         elif cmd == 'reset':
    #             obs, joint_map = env.reset()
    #             remote.send((obs, joint_map))
    #         elif cmd == 'close':
    #             remote.close()
    #             break
    #         else:
    #             raise NotImplementedError

    def worker(self, remote, parent_remote, env):
        parent_remote.close()

        while True:
            cmd, data = remote.recv()
            if cmd == 'step':

                obs, reward, done, info, joint_map, rescue_masks = env.step(data)
                if done[0]:
                    obs, joint_map = env.reset()
                remote.send((obs, reward, done, info, joint_map, rescue_masks))
            elif cmd == 'reset':
                obs, joint_map = env.reset()
                remote.send((obs, joint_map))
            elif cmd == 'close':
                remote.close()
                break
            else:
                raise NotImplementedError



