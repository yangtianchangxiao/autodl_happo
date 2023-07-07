import numpy as np
import torch
import torch.nn as nn
from utils.util import get_gard_norm, huber_loss, mse_loss
from utils.popart import PopArt
from algorithms.utils.util import check

class HAPPO():
    """
    Trainer class for HAPPO to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (HAPPO_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self,
                 args,
                 policy,
                 device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm       
        self.huber_delta = args.huber_delta

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks

        
        if self._use_popart:
            self.value_normalizer = PopArt(1, device=self.device)
        else:
            self.value_normalizer = None
        self.batch_expand_times = args.batch_expand_times
        error_message = f"{self.num_mini_batch} should be greater than {self.batch_expand_times}"
        # assert self.num_mini_batch >= self.batch_expand_times, error_message
        
        self.train_times = 1
        
    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead at a given timesep.

        :return value_loss: (torch.Tensor) value function loss.
        """
        if self._use_popart:
            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                        self.clip_param)
            error_clipped = self.value_normalizer(return_batch) - value_pred_clipped
            error_original = self.value_normalizer(return_batch) - values
        else:
            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                        self.clip_param)
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def ppo_update(self, sample, sample_time, update_actor=True):
        """
        Update actor and critic networks.
        :param sample: (Tuple) contains data batch with which to update networks.
        :update_actor: (bool) whether to update actor network.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic update.
        ;return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
        value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, available_actions_batch, factor_batch, rescue_masks_batch = sample

        

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)


        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)


        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        factor_batch = check(factor_batch).to(**self.tpdv)
        # Reshape to do in a single forward pass for all steps
        # 增加一个 intervention_mask, 当 mask 的某个元素是 0 时，loss 是 behavior clone loss; 当该元素是 1 时，loss 是 happo 的 loss
        values, action_log_probs, dist_entropy, soft_probs_batch = self.policy.evaluate_actions(share_obs_batch,
                                                                              obs_batch, 
                                                                              rnn_states_batch, 
                                                                              rnn_states_critic_batch, 
                                                                              actions_batch, 
                                                                              masks_batch, 
                                                                              available_actions_batch,
                                                                              active_masks_batch)
        # actor update
        imp_weights = torch.prod(torch.exp(action_log_probs - old_action_log_probs_batch),dim=-1,keepdim=True)

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
        # print("surr1: ",surr1, "surr2", surr2)
        # 原本 rescue masks 里由-1 -2 -3 -4， 和 1，1表示用的是policy的action，其余的表示用的是rescue 的action
        # 所以我们先把所有的负数变成 0， 然后得到纯policy部分的 error
        policy_masks = np.clip(rescue_masks_batch, 0, None)
        if self._use_policy_active_masks:
            policy_masks_tensor = torch.tensor(policy_masks, dtype=torch.float32, device=factor_batch.device)

            # policy_action_loss = (-torch.sum(policy_masks_tensor * factor_batch * torch.min(surr1, surr2),
            policy_action_loss = (-torch.sum(factor_batch * torch.min(surr1, surr2),
                                             dim=-1,
                                             keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(factor_batch * torch.min(surr1, surr2), dim=-1, keepdim=True).mean()



        # modified_masks = np.clip(-rescue_masks_batch, 0, None)
        # non_zero_indices = np.nonzero(modified_masks)[0]
        # intervention_loss = torch.sum(soft_probs_batch[non_zero_indices, modified_masks[non_zero_indices]- 1])
        # intervention_loss = sum(soft_probs_batch[idx + modified_masks[idx] - 1] for idx in non_zero_indices)

        # 但是需要注意下，interbention_loss是否有梯度？
        
        # print("policy_loss is ", policy_action_loss, " and intervention_loss is ", intervention_loss)
        # policy_loss = policy_action_loss + 0.00001*intervention_loss
        policy_loss = policy_action_loss
        if self.batch_expand_times > 1:
            policy_loss = policy_loss / self.batch_expand_times
        if update_actor:
            (policy_loss - dist_entropy * self.entropy_coef).backward()


        # 当样本拓展一定倍数后，更新actor
        
        # if sample_time % self.batch_expand_times ==0 or sample_time == self.num_mini_batch:
        if self.train_times % self.batch_expand_times ==0:
            if self._use_max_grad_norm:
                actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
            else:
                actor_grad_norm = get_gard_norm(self.policy.actor.parameters())
            self.policy.actor_optimizer.step()
            self.policy.actor_optimizer.zero_grad()

        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)
        if self.batch_expand_times > 1:
            ((value_loss * self.value_loss_coef)/ self.batch_expand_times).backward()
        
        # 当样本拓展一定倍数后，更新critic
        # if sample_time % self.batch_expand_times ==0 or sample_time == self.num_mini_batch:
        if self.train_times % self.batch_expand_times == 0:
            if self._use_max_grad_norm:
                critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
            else:
                critic_grad_norm = get_gard_norm(self.policy.critic.parameters())
            self.policy.critic_optimizer.step()
            self.policy.critic_optimizer.zero_grad()

            # 更新训练次数
            self.train_times = 1
        else:
            self.train_times += 1

        # print("train_times is", self.train_times)
        # return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights

    def train(self, buffer, update_actor=True):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        # 在原本的 buffer 中加入 mask，计算 advantage 时，returns 和 value_preds 只用 mask 中 值为 1 的元素所在的index
        if self._use_popart:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]

        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        train_info = {}
        time = 0
        # train_info['value_loss'] = 0
        # train_info['policy_loss'] = 0
        # train_info['dist_entropy'] = 0
        # train_info['actor_grad_norm'] = 0
        # train_info['critic_grad_norm'] = 0
        # train_info['ratio'] = 0
        if self._use_recurrent_policy:
            data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length)
        elif self._use_naive_recurrent:
            data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch)
        else:
            data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)

        for _ in range(self.ppo_epoch):
            for sample_time, sample in enumerate(data_generator):
                # value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights = self.ppo_update(sample, update_actor=update_actor)
                # sample_time 从1开始计数，这样可以避免一开始sample_time % batch_expand_time ==0
                self.ppo_update(sample, sample_time+1, update_actor=update_actor)
                # 显示显存使用情况
                # print('Current GPU memory usage:')
                # print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
                # print('Reserved: ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
                # time = time + 1 
                # print("time is ", time)
        num_updates = self.ppo_epoch * self.num_mini_batch
        
        for k in train_info.keys():
            train_info[k] /= num_updates
            
        return train_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()
