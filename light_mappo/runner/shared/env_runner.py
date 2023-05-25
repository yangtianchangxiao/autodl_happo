"""
# @Time    : 2021/7/1 7:15 下午
# @Author  : hezhiqiang01
# @Email   : hezhiqiang01@baidu.com
# @File    : env_runner.py
"""

import time
import numpy as np
import torch
import datetime
# import taichi
from light_mappo.runner.shared.base_runner import Runner
# import imageio
from tensorboardX import SummaryWriter


def _t2n(x):
    if type(x) is tuple:
        x_np = tuple(elem.detach().cpu().numpy() for elem in x)
        return x_np
    else:
        return x.detach().cpu().numpy()



class EnvRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for the MPEs. See parent class for details."""

    def __init__(self, config, log_dir_address = 'D:/code/light_mappo_test/test13'):
        super(EnvRunner, self).__init__(config)
        self.log_dir_address = log_dir_address
        self.reset_count = 0
    def run(self):
        # print("self.envs.action_space[0].__class__.__name__",self.envs.action_space[0].__class__.__name__)
        print("share_observation_space: ", self.envs.share_observation_space)
        print("observation_space: ", self.envs.observation_space)
        print("action_space: ", self.envs.action_space)

        self.warmup()

        start = time.time()
        print("start time is",datetime.datetime.now())
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        target_find_list = []
        reward_list = []
        # load_dir = '/home/cx/mappo_model/cnn_run55/'
        # checkpoint_actor = torch.load(
        #     load_dir+'actor.pt')
        # checkpoint_critic = torch.load(
        #     load_dir+'critic.pt')
        # checkpoint_acotr_opti = torch.load(load_dir+'actor_adma.pt')
        # checkpoint_critic_opti = torch.load(load_dir+'critic_adma.pt')
        # self.trainer.policy.actor.load_state_dict(checkpoint_actor)
        # self.trainer.policy.critic.load_state_dict(checkpoint_critic)
        # self.trainer.policy.actor_optimizer.load_state_dict(checkpoint_acotr_opti)
        # self.trainer.policy.critic_optimizer.load_state_dict(checkpoint_critic_opti)

        print("episodes are", episodes)
        last_average_reward = -1000
        best_episode = 0
        for episode in range(episodes):
            print("episode is", episode)
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)
            self.reset_count = 0
            for step in range(self.episode_length):
                # Sample actions
                # actions_logits_value is the log probs of all actions while action_log_probs is the single one
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env, actions_logits_value = self.collect(step)

                # Obser reward and next obs
                obs, rewards, dones, infos, joint_map = self.envs.step(actions_env)
                # print("dones", dones.shape, type(dones))
                # Extract the first column
                first_column = dones[:, 0]

                # Count the number of True values in the first column
                count_true = np.count_nonzero(first_column)
                self.reset_count = self.reset_count + count_true

                for i, row in enumerate(dones):
                    if True in row:
                        if infos[i] >= 0:

                            target_find_list.append(infos[i])
                for i in rewards:
                    reward_list.append(np.mean(i))
                # print("reward list is",reward_list)
                # print("target_find_list",target_find_list)
                # print("rewards is",rewards)
                # print("done is",dones)

                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic, joint_map, actions_logits_value, actions_env
                # insert data into buffer

                self.insert(data)

            # compute return and update network
            self.compute()
            print("Start train")
            train_infos = self.train()
            print("End train")

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads



            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}. Reset {} times.\n"
                      .format(self.all_args.scenario_name,
                              self.algorithm_name,
                              self.experiment_name,
                              episode,
                              episodes,
                              total_num_steps,
                              self.num_env_steps,
                              int(total_num_steps / (end - start)),
                              self.reset_count))
                now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                print("time is", now)
                # if self.env_name == "MPE":
                #     env_infos = {}
                #     for agent_id in range(self.num_agents):
                #         idv_rews = []
                #         for info in infos:
                #             if 'individual_reward' in info[agent_id].keys():
                #                 idv_rews.append(info[agent_id]['individual_reward'])
                #         agent_k = 'agent%i/individual_rewards' % agent_id
                #         env_infos[agent_k] = idv_rews

                # train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                train_infos["average_episode_rewards"] = np.mean(self.buffer.original_rewards)
                average_total_reward = np.mean(self.buffer.rewards)
                print("average episode extrinsic reward is {}".format(train_infos["average_episode_rewards"]))
                # print("average episode intrinsic reward is {}".format(average_total_reward - train_infos["average_episode_rewards"]))
                print("average episode whole reward is {}".format(average_total_reward))
                # print("s",self.l)
                # self.log_train(train_infos, total_num_steps)
                # self.log_env(env_infos, total_num_steps)

                # save model
                if (episode % self.save_interval == 0 or episode == episodes - 1):
                    if train_infos["average_episode_rewards"] > last_average_reward:
                        print("save the model in episode", episode)
                        best_episode = episode
                        self.save()
                        last_average_reward = train_infos["average_episode_rewards"]
                    else:
                        print("Not better than episode", best_episode, ". And the the highest reward is", last_average_reward)
                        self.save_copy()

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)
        with SummaryWriter(log_dir=self.log_dir_address, comment='Target found') as w:
            for i in range(len(target_find_list)):
                w.add_scalar('Targets found during training', target_find_list[i], i)
            for i in range(len(reward_list)):
                w.add_scalar('The change of reward', reward_list[i], i)
            print("已经保存",self.log_dir_address)
        w.close()
        print("self.log_dir_address", self.log_dir_address)

    def warmup(self):
        # reset env
        obs, joint_map = self.envs.reset()  # shape = (5, 2, 14)

        # replay buffer
        if self.use_centralized_V:
            # share_obs = obs.reshape(self.n_rollout_threads, -1)  # shape = (5, 28)
            # share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)  # shape = (5, 2, 28)
            # 当使用mlp时，用这个
            # share_obs = joint_map.reshape(self.n_rollout_threads, -1)  # shape = (5, 28)
            # 当使用CNN时，用这个
            share_obs = joint_map  # shape = (5, 28)
            # share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)  # shape = (5, 2, 28)

        else:
            share_obs = obs

        # 这是mlp时候用的
        # self.buffer.share_obs[0] = share_obs.copy().reshape(20,-1,7500)
        self.buffer.share_obs[0] = share_obs.copy()

        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        # print("np.concatenate(self.buffer.rnn_states[step]) shape",np.concatenate(self.buffer.rnn_states[step]).shape)
        # value is the expected Q.

        value, action, action_log_prob, rnn_states, rnn_states_critic, actions_logits_value \
            = self.trainer.policy.get_actions(self.buffer.share_obs[step],
                                              np.concatenate(self.buffer.obs[step]),
                                              np.concatenate(self.buffer.rnn_states[step]),
                                              np.concatenate(self.buffer.rnn_states_critic[step]),
                                              np.concatenate(self.buffer.masks[step]))
        # [self.envs, agents, dim]

        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        # shape is 20,2,1
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))

        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))


        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        actions_logits_value = np.array(np.split(_t2n(actions_logits_value), self.n_rollout_threads))

        # rearrange action
        if self.envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
            for i in range(self.envs.action_space[0].shape):
                uc_actions_env = np.eye(self.envs.action_space[0].high[i] + 1)[actions[:, :, i]]
                if i == 0:
                    actions_env = uc_actions_env
                else:
                    actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
        elif self.envs.action_space[0].__class__.__name__ == 'Discrete':
            # actions  --> actions_env : shape:[10, 1] --> [5, 2, 5]
            # print("actions are", actions.shape)
            actions_env = np.squeeze(np.eye(self.envs.action_space[0].n,  dtype=np.float32)[actions], 2)
        else:
            # TODO 这里改造成自己环境需要的形式即可
            actions_env = actions
            # raise NotImplementedError
        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env, actions_logits_value

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic, joint_map, actions_logits_value, actions_env = data
        # In mappo, dones is the same with it in rmappo. So I guess it is the rnn_states is a tuple of two elements that causes the error
        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size),
                                              dtype=np.float32)
        # rnn_states[dones == True] = (np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size),
        #                                      dtype=np.float32),
        #                              np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size),
        #                                      dtype=np.float32))


        critic_dones = dones[:,0]

        rnn_states_critic[critic_dones == True] = np.zeros(((critic_dones == True).sum(), *self.buffer.rnn_states_critic.shape[2:]),
                                                     dtype=np.float32)
        # rnn_states_critic[dones == True] = (np.zeros(((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]),
        #                                             dtype=np.float32),
        #                                     np.zeros(((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]),
        #                                             dtype=np.float32))



        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        if self.use_centralized_V:
            # share_obs = obs.reshape(self.n_rollout_threads, -1)
            # share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
            # 当使用mlp时，用这个
            # share_obs = joint_map.reshape(self.n_rollout_threads, -1)
                        # share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
                        # print("share obs shape centra",share_obs.shape)
            # 当使用CNN时，用这个
            share_obs = joint_map  # shape = (5, 28)

        else:
            share_obs = obs
            # print("share obs shape",share_obs.shape)

        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards,
                           masks, actions_logits_value, actions_env)

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            self.trainer.prep_rollout()
            eval_action, eval_rnn_states = self.trainer.policy.act(np.concatenate(eval_obs),
                                                                   np.concatenate(eval_rnn_states),
                                                                   np.concatenate(eval_masks),
                                                                   deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

            if self.eval_envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                for i in range(self.eval_envs.action_space[0].shape):
                    eval_uc_actions_env = np.eye(self.eval_envs.action_space[0].high[i] + 1)[eval_actions[:, :, i]]
                    if i == 0:
                        eval_actions_env = eval_uc_actions_env
                    else:
                        eval_actions_env = np.concatenate((eval_actions_env, eval_uc_actions_env), axis=2)
            elif self.eval_envs.action_space[0].__class__.__name__ == 'Discrete':
                eval_actions_env = np.squeeze(np.eye(self.eval_envs.action_space[0].n)[eval_actions], 2)
            else:
                raise NotImplementedError

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_env_infos = {}
        eval_env_infos['eval_average_episode_rewards'] = np.sum(np.array(eval_episode_rewards), axis=0)
        eval_average_episode_rewards = np.mean(eval_env_infos['eval_average_episode_rewards'])
        print("eval average episode rewards of agent: " + str(eval_average_episode_rewards))
        self.log_env(eval_env_infos, total_num_steps)

    @torch.no_grad()
    def render(self):
        """Visualize the env."""
        envs = self.envs

        all_frames = []
        for episode in range(self.all_args.render_episodes):
            obs = envs.reset()
            if self.all_args.save_gifs:
                image = envs.render('rgb_array')[0][0]
                all_frames.append(image)
            else:
                envs.render('human')

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size),
                                  dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

            episode_rewards = []

            for step in range(self.episode_length):
                calc_start = time.time()

                self.trainer.prep_rollout()
                action, rnn_states = self.trainer.policy.act(np.concatenate(obs),
                                                             np.concatenate(rnn_states),
                                                             np.concatenate(masks),
                                                             deterministic=True)
                actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))

                if envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                    for i in range(envs.action_space[0].shape):
                        uc_actions_env = np.eye(envs.action_space[0].high[i] + 1)[actions[:, :, i]]
                        if i == 0:
                            actions_env = uc_actions_env
                        else:
                            actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
                elif envs.action_space[0].__class__.__name__ == 'Discrete':
                    actions_env = np.squeeze(np.eye(envs.action_space[0].n)[actions], 2)
                else:
                    raise NotImplementedError

                # Obser reward and next obs
                obs, rewards, dones, infos, joint_map = envs.step(actions_env)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size),
                                                     dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    image = envs.render('rgb_array')[0][0]
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)
                else:
                    envs.render('human')

            print("average episode rewards is: " + str(np.mean(np.sum(np.array(episode_rewards), axis=0))))

        # if self.all_args.save_gifs:
        #     imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames, duration=self.all_args.ifi)
