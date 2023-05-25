import os
import numpy as np
import torch
from tensorboardX import SummaryWriter
from light_mappo.utils.shared_buffer import SharedReplayBuffer

def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()

class Runner(object):
    """
    Base class for training recurrent policies.
    :param config: (dict) Config dictionary containing parameters for training.
    """
    def __init__(self, config):

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']
        if config.__contains__("render_envs"):
            self.render_envs = config['render_envs']       

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.n_render_rollout_threads = self.all_args.n_render_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir

        self.run_dir = config["run_dir"]
        self.log_dir = str(self.run_dir / 'logs')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.writter = SummaryWriter(self.log_dir)
        self.save_dir = str(self.run_dir / 'models')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        from light_mappo.algorithms.algorithm.r_mappo import RMAPPO as TrainAlgo
        from light_mappo.algorithms.algorithm.rMAPPOPolicy import RMAPPOPolicy as Policy

        share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0]
        # policy network
        self.ICMModel = None
        if self.all_args.enable_ICM:
            from light_mappo.curiosity.icm import MlpICMModel

            self.ICMModel = MlpICMModel(self.all_args, self.envs.observation_space[0].shape[0],
                                        self.envs.action_space[0].n, device=self.device)
        # print("self.envs.observation_space", self.envs.observation_space[0], self.envs.observation_space[0].shape)
        # 由于智能体数量大于一，所以observation_space 和  action_space 都是[box1, box2, ...]
        self.policy = Policy(self.all_args,
                            self.envs.observation_space[0],
                            share_observation_space,
                            self.envs.action_space[0],
                            icm_module= self.ICMModel,
                            device = self.device)

        if self.model_dir is not None:
            self.restore()

        # algorithm
        self.trainer = TrainAlgo(self.all_args, self.policy, device = self.device)

        # buffer
        # self.buffer = SharedReplayBuffer(self.all_args,
        #                                 self.num_agents,
        #                                 self.envs.observation_space[0],
        #                                 share_observation_space,
        #                                 self.envs.action_space[0])
        self.buffer = SharedReplayBuffer(self.all_args,
                                         self.num_agents,
                                         self.envs.observation_space,
                                         share_observation_space,
                                         self.envs.action_space[0])
    def run(self):
        """Collect training data, perform training updates, and evaluate policy."""
        raise NotImplementedError

    def warmup(self):
        """Collect warmup pre-training data."""
        raise NotImplementedError

    def collect(self, step):
        """Collect rollouts for training."""
        raise NotImplementedError

    def insert(self, data):
        """
        Insert data into buffer.
        :param data: (Tuple) data to insert into training buffer.
        """
        raise NotImplementedError
    
    # @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        self.trainer.prep_rollout()
        # 如果使用mlp，用这个
        # next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
        #                                         np.concatenate(self.buffer.rnn_states_critic[-1]),
        #                                         np.concatenate(self.buffer.masks[-1]))
        # 如果使用cnn, 用这个
        next_values = self.trainer.policy.get_values(self.buffer.share_obs[-1],
                                                np.concatenate(self.buffer.rnn_states_critic[-1]),
                                                np.concatenate(self.buffer.masks[-1]))
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))

        # calculate self.buffer.rewards

        # In MLPICMModel the forward function is: def forward(self, inputs):
        # And state, next_state, action = inputs
        if self.all_args.enable_ICM:
            pred_next_state_feature_orig, encode_next_state, pred_action = self.trainer.policy.icm_module((
                self.buffer.obs[:-1], self.buffer.obs[1:], self.buffer.actions_env))
            # print("self.buffer.obs",self.buffer.obs.shape)
            # print("pred_next_state_feature_orig shape",pred_next_state_feature_orig.shape)
            # self.buffer.pred_next_state_feature_orig = pred_next_state_feature_orig
            # self.buffer.encode_next_state = encode_next_state
            # self.buffer.pred_action = pred_action

            rewards = self.trainer.policy.icm_module.rewards(pred_next_state_feature_orig, encode_next_state)
            # print("reward sahpe", rewards.shape)
            # print("self reward shap", self.buffer.rewards.shape)
            # rewards = np.array(np.split(_t2n(rewards), self.n_rollout_threads))
            # We get rewards as a tensor but tensor data cannot add with array directly, so we convert tensor to array
            # print("Extrinsic reward is", self.buffer.rewards[0])
            self.buffer.original_rewards = self.buffer.rewards.copy()
            # print("Intrinsic reward is", rewards[0])
            self.buffer.rewards += rewards.numpy()
        else:
            self.buffer.original_rewards = self.buffer.rewards.copy()
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)
    
    def train(self):
        """Train policies with data in buffer. """
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer)      
        self.buffer.after_update()
        return train_infos

    def save(self):
        """Save policy's actor and critic networks."""
        # 这其实是con run 22 只不过输入弄错了好像
        # model_log_dir_address = "/home/cx/mappo_model/cnn_ICMrun_02/"
        model_log_dir_address = "/home/cx/mappo_model/cnn_run57_12/"
        print("save dir is", model_log_dir_address)
        # model_log_dir_address = "/home/cx/env_test_save/"
        if not os.path.exists(model_log_dir_address):
            print("不存在该路径，正在创建")
            os.makedirs(model_log_dir_address)
        policy_actor = self.trainer.policy.actor
        torch.save(policy_actor.state_dict(),  model_log_dir_address + "actor.pt")
        policy_critic = self.trainer.policy.critic
        torch.save(policy_critic.state_dict(), model_log_dir_address + "critic.pt")
        actor_optimizer = self.trainer.policy.actor_optimizer
        torch.save(actor_optimizer.state_dict(),model_log_dir_address + "actor_adma.pt")
        critic_optimizer = self.trainer.policy.critic_optimizer
        torch.save(critic_optimizer.state_dict(), model_log_dir_address + "critic_adma.pt")

        # 保存 actor critic actor_adma critic_adma actor_norm critic_norm
        # policy_actor = self.trainer.policy.actor
        # policy_critic = self.trainer.policy.critic
        # actor_optimizer = self.trainer.policy.actor_optimizer
        # critic_optimizer = self.trainer.policy.critic_optimizer
        #
        # actor_state_dict = policy_actor.state_dict()
        # critic_state_dict = policy_critic.state_dict()
        #
        # actor_norm_mean = policy_actor.layer_norm.running_mean
        # actor_norm_var = policy_actor.layer_norm.running_var
        # actor_state_dict['layer_norm.running_mean'] = actor_norm_mean
        # actor_state_dict['layer_norm.running_var'] = actor_norm_var
        #
        # critic_norm_mean = policy_critic.layer_norm.running_mean
        # critic_norm_var = policy_critic.layer_norm.running_var
        # critic_state_dict['layer_norm.running_mean'] = critic_norm_mean
        # critic_state_dict['layer_norm.running_var'] = critic_norm_var
        #
        # torch.save(actor_state_dict, model_log_dir_address + "actor.pt")
        # torch.save(critic_state_dict, model_log_dir_address + "critic.pt")
        # torch.save(actor_optimizer.state_dict(),model_log_dir_address + "actor_adma.pt")
        # torch.save(critic_optimizer.state_dict(), model_log_dir_address + "critic_adma.pt")


    def save_copy(self):
        """Save policy's actor and critic networks."""
        # 这其实是con run 22 只不过输入弄错了好像
        # model_log_dir_address = "/home/cx/mappo_model/cnn_ICMrun_02/"
        model_log_dir_address = "/home/cx/mappo_model/cnn_run57_12/"
        print("save dir is", model_log_dir_address)
        # model_log_dir_address = "/home/cx/env_test_save/"
        if not os.path.exists(model_log_dir_address):
            print("不存在该路径，正在创建")
            os.makedirs(model_log_dir_address)
        policy_actor = self.trainer.policy.actor
        torch.save(policy_actor.state_dict(),  model_log_dir_address + "actor_copy.pt")


    def restore(self):
        """Restore policy's networks from a saved model."""
        policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor.pt')
        self.policy.actor.load_state_dict(policy_actor_state_dict)
        if not self.all_args.use_render:
            policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic.pt')
            self.policy.critic.load_state_dict(policy_critic_state_dict)
 
    def log_train(self, train_infos, total_num_steps):
        """
        Log training info.
        :param train_infos: (dict) information about training update.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in train_infos.items():
            self.writter.add_scalars(k, {k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        """
        Log env info.
        :param env_infos: (dict) information about env state.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in env_infos.items():
            if len(v)>0:
                self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
