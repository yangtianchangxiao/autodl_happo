import time
import numpy as np
from functools import reduce
import torch
from runners.separated.base_runner import Runner
from tensorboardX import SummaryWriter

def _t2n(x):
    return x.detach().cpu().numpy()

class SMACRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""
    def __init__(self, config, log_dir_address):
        super(SMACRunner, self).__init__(config, log_dir_address)
        self.log_dir_address = log_dir_address

    def run(self):
        print("start run")


        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        last_battles_game = np.zeros(self.n_rollout_threads, dtype=np.float32)
        last_battles_won = np.zeros(self.n_rollout_threads, dtype=np.float32)

        print("episodes are", episodes)
        last_average_reward = -1000
        best_episode = 0

        target_find_list = []
        with SummaryWriter(log_dir=self.log_dir_address, comment='Target found') as w:
            last_episode = 0
            for episode in range(episodes):
                self.reset_count = 0
                self.reset_count2 = 0
                if self.use_linear_lr_decay:
                    self.trainer.policy.lr_decay(episode, episodes)

                for step in range(self.episode_length):
                    # Sample actions
                    values, actions, action_log_probs, rnn_states, rnn_states_critic, soft_probs = self.collect(step)
                    # Obser reward and next obs
                    obs,  rewards, dones, infos, share_obs, rescue_masks = self.envs.step(actions)
                    # print("rewards", rewards)
                    data = obs, share_obs, rewards, dones, infos, \
                           values, actions, action_log_probs, \
                           rnn_states, rnn_states_critic, soft_probs, rescue_masks

                    # insert data into buffer
                    self.insert(data)
                    # Extract the first column
                    first_column = dones[:, 0]
                    second_column = dones[:, 1]
                    # print("dones type is", type(dones), dones[:,0].shape)
                    # Count the number of True values in the first column
                    count_true = np.count_nonzero(first_column)
                    count_true2 = np.count_nonzero(second_column)
                    self.reset_count = self.reset_count + count_true
                    self.reset_count2 = self.reset_count2 + count_true2
                    for i, row in enumerate(dones):
                        if True in row:
                            if infos[i] >= 0:
                                target_find_list.append(infos[i])
                # compute return and update network
                self.compute()
                # print("start training")
                train_infos = self.train()

                # post process
                total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
                average_total_reward = self.average_rewards()
                # average_total_reward = np.mean(self.buffer.rewards)

                # save model
                if (episode % self.save_interval == 0 or episode == episodes - 1):
                    # print("average reward", average_total_reward)
                    # print("last_average reawrd", last_average_reward)
                    if average_total_reward > last_average_reward:
                        print("save the model in episode", episode)
                        best_episode = episode
                        self.save()
                        last_average_reward = average_total_reward
                    else:
                        print("Not better than episode", best_episode, ". And the the highest reward is",
                              last_average_reward)
                        self.save_copy()


                # log information
                if episode % self.log_interval == 0:
                    end = time.time()
                    print("\n Map {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}. Reward is{}. Reset time1 is{}, reset time2 is {}\n"
                            .format(self.all_args.map_name,
                                    self.algorithm_name,
                                    self.experiment_name,
                                    episode,
                                    episodes,
                                    total_num_steps,
                                    self.num_env_steps,
                                    int(total_num_steps / (end - start)),
                                    average_total_reward,
                                    self.reset_count,
                                    self.reset_count2))
                    self.reset_count = 0
                    self.reset_count2 = 0
                    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    print("time is", now)

                    # print("average episode extrinsic reward is {}".format(train_infos["average_episode_rewards"]))
                    # print("average episode intrinsic reward is {}".format(average_total_reward - train_infos["average_episode_rewards"]))
                    print("average episode whole reward is {}".format(average_total_reward))
                    if self.env_name == "StarCraft2":
                        battles_won = []
                        battles_game = []
                        incre_battles_won = []
                        incre_battles_game = []

                        for i, info in enumerate(infos):
                            if 'battles_won' in info[0].keys():
                                battles_won.append(info[0]['battles_won'])
                                incre_battles_won.append(info[0]['battles_won']-last_battles_won[i])
                            if 'battles_game' in info[0].keys():
                                battles_game.append(info[0]['battles_game'])
                                incre_battles_game.append(info[0]['battles_game']-last_battles_game[i])

                        incre_win_rate = np.sum(incre_battles_won)/np.sum(incre_battles_game) if np.sum(incre_battles_game)>0 else 0.0
                        print("incre win rate is {}.".format(incre_win_rate))
                        self.writter.add_scalars("incre_win_rate", {"incre_win_rate": incre_win_rate}, total_num_steps)

                        last_battles_game = battles_game
                        last_battles_won = battles_won
                    # modified

                    for agent_id in range(self.num_agents):
                        train_infos[agent_id]['dead_ratio'] = 1 - self.buffer[agent_id].active_masks.sum() /(self.num_agents* reduce(lambda x, y: x*y, list(self.buffer[agent_id].active_masks.shape)))

                    self.log_train(train_infos, total_num_steps)

                # eval
                if episode % self.eval_interval == 0 and self.use_eval:
                    self.eval(total_num_steps)

                # 写入数据
                for i in range(len(target_find_list)):
                    w.add_scalar('Targets found during training', target_find_list[i], i+last_episode)
                last_episode = last_episode + len(target_find_list)
                target_find_list = []
                print("已经保存",self.log_dir_address)
        w.close()
        print("self.log_dir_address", self.log_dir_address)

    def warmup(self):
        # reset env
        obs, share_obs = self.envs.reset()
        # replay buffer
        if not self.use_centralized_V:
            share_obs = obs
        for agent_id in range(self.num_agents):
            # print("share_obs shape is", share_obs.shape)
            # print("self.buffer[agent_id].share_obs[0] shape is ", self.buffer[agent_id].share_obs[0].shape)
            self.buffer[agent_id].share_obs[0] = share_obs.copy()
            # print("obs shape", obs.shape)
            self.buffer[agent_id].obs[0] = obs[:,agent_id].copy()

    @torch.no_grad()
    def collect(self, step, soft_probs = None):
        value_collector=[]
        action_collector=[]
        action_log_prob_collector=[]
        rnn_state_collector=[]
        rnn_state_critic_collector=[]
        soft_prob_collector=[]
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()

            value, action, action_log_prob, rnn_state, rnn_state_critic, soft_prob \
                = self.trainer[agent_id].policy.get_actions(self.buffer[agent_id].share_obs[step],
                                                self.buffer[agent_id].obs[step],
                                                self.buffer[agent_id].rnn_states[step],
                                                self.buffer[agent_id].rnn_states_critic[step],
                                                self.buffer[agent_id].masks[step])
            # print("share_obs[step].shape", self.buffer[agent_id].share_obs[step].shape)
            # print("obs[step] shape", self.buffer[agent_id].obs[step].shape)
            #
            # print("self.buffer[agent_id].masks[step]",self.buffer[agent_id].masks[step].shape)
            # print("self.buffer[agent_id].rnn_states[step]", self.buffer[agent_id].rnn_states[step].shape)
            # print("self.buffer[agent_id].rnn_states_critic[step]", self.buffer[agent_id].rnn_states_critic[step].shape)
            # print("actions", action)
            # print("")
            value_collector.append(_t2n(value))
            action_collector.append(_t2n(action))
            action_log_prob_collector.append(_t2n(action_log_prob))
            rnn_state_collector.append(_t2n(rnn_state))
            soft_prob_collector.append(_t2n(soft_prob))
            rnn_state_critic_collector.append(_t2n(rnn_state_critic))
        # [self.envs, agents, dim]
        values = np.array(value_collector).transpose(1, 0, 2)
        # print("action_collector are",action_collector)

        actions = np.array(action_collector).transpose(1, 0, 2)
        # print("actions are", action_collector)
        action_log_probs = np.array(action_log_prob_collector).transpose(1, 0, 2)
        rnn_states = np.array(rnn_state_collector).transpose(1, 0, 2, 3)
        rnn_states_critic = np.array(rnn_state_critic_collector).transpose(1, 0, 2, 3)
        soft_probs = np.array(soft_prob_collector).transpose(1, 0, 2)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, soft_probs

    def insert(self, data):
        obs, share_obs, rewards, dones, infos, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic, soft_probs, rescue_mask = data
        # print("actions is", actions)
        dones_env = np.array(dones)
        # dones_env = np.all(dones, axis=1)

        # rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        # rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, *self.buffer[0].rnn_states_critic.shape[2:]), dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        # print("dones env .shape", dones_env.shape)
        # masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        # active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        # active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        # active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        # bad_masks = np.array([[[0.0] if info[agent_id]['bad_transition'] else [1.0] for agent_id in range(self.num_agents)] for info in infos])
        
        if not self.use_centralized_V:
            share_obs = obs
        for agent_id in range(self.num_agents):
            self.buffer[agent_id].insert(share_obs[:,agent_id], obs[:,agent_id], rnn_states[:,agent_id],
                    rnn_states_critic[:,agent_id],actions[:,agent_id], action_log_probs[:, agent_id],
                    values[:,agent_id], rewards[:,agent_id], masks[:,agent_id], soft_probs[:, agent_id], rescue_mask[:, agent_id])

    def average_rewards(self):
        total_rewards = 0

        for agent_id in range(self.num_agents):
            total_rewards += sum(self.buffer[agent_id].rewards)
        total_rewards = np.sum(total_rewards)

        average_reward = total_rewards / self.num_agents/self.n_rollout_threads/self.episode_length
        print("return average reward", average_reward)
        return average_reward
    def log_train(self, train_infos, total_num_steps):
        for agent_id in range(self.num_agents):
            train_infos[agent_id]["average_step_rewards"] = np.mean(self.buffer[agent_id].rewards)
            for k, v in train_infos[agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)
    
    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_battles_won = 0
        eval_episode = 0

        eval_episode_rewards = []
        one_episode_rewards = []
        for eval_i in range(self.n_eval_rollout_threads):
            one_episode_rewards.append([])
            eval_episode_rewards.append([])

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        while True:
            eval_actions_collector=[]
            eval_rnn_states_collector=[]
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()
                eval_actions, temp_rnn_state = \
                    self.trainer[agent_id].policy.act(eval_obs[:,agent_id],
                                            eval_rnn_states[:,agent_id],
                                            eval_masks[:,agent_id],
                                            eval_available_actions[:,agent_id],
                                            deterministic=True)
                eval_rnn_states[:,agent_id]=_t2n(temp_rnn_state)
                eval_actions_collector.append(_t2n(eval_actions))

            eval_actions = np.array(eval_actions_collector).transpose(1,0,2)

            
            # Obser reward and next obs
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.eval_envs.step(eval_actions)
            for eval_i in range(self.n_eval_rollout_threads):
                one_episode_rewards[eval_i].append(eval_rewards[eval_i])

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)

            eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards[eval_i].append(np.sum(one_episode_rewards[eval_i], axis=0))
                    one_episode_rewards[eval_i] = []
                    if eval_infos[eval_i][0]['won']:
                        eval_battles_won += 1

            if eval_episode >= self.all_args.eval_episodes:
                eval_episode_rewards = np.concatenate(eval_episode_rewards)
                eval_env_infos = {'eval_average_episode_rewards': eval_episode_rewards}                
                self.log_env(eval_env_infos, total_num_steps)
                eval_win_rate = eval_battles_won/eval_episode
                print("eval win rate is {}.".format(eval_win_rate))
                self.writter.add_scalars("eval_win_rate", {"eval_win_rate": eval_win_rate}, total_num_steps)
                break
