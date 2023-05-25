

import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import math
from light_mappo.utils.util import get_gard_norm, huber_loss, mse_loss
import torch.nn.functional as F
from light_mappo.algorithms.utils.util import init as init_module, get_clones
from torch.nn import init


class Swish(nn.Module):
    def forward(self, x):
        return x * F.sigmoid(x)


def swish(x):
    return x * F.sigmoid(x)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class MlpActorCriticNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(MlpActorCriticNetwork, self).__init__()
        self.com_layer1 = nn.Linear(input_size, 256)
        self.batch_1 = nn.BatchNorm1d(256)
        self.com_layer2 = nn.Linear(256, 256)
        self.batch_2 = nn.BatchNorm1d(256)
        self.com_layer3 = nn.Linear(256, 256)
        self.batch_3 = nn.BatchNorm1d(256)

        self.actor_1 = nn.Linear(256, 256)
        self.actor_2 = nn.Linear(256, 256)
        self.actor = nn.Linear(256, output_size)
        self.critic_1 = nn.Linear(256, 256)
        self.critic_2 = nn.Linear(256, 256)
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        x = swish(self.batch_1(self.com_layer1(x)))
        x = swish(self.batch_2(self.com_layer2(x)))
        x = swish(self.batch_3(self.com_layer3(x)))
        actor_1 = swish(self.actor_1(x))
        actor_2 = swish(self.actor_2(x))
        policy = self.actor(actor_2)
        critic_1 = swish(self.critic_1(x))
        critic_2 = swish(self.critic_2(x))
        value = self.critic(critic_2)

        return policy, value


class MlpICMModel(nn.Module):
    # input_dim is obs.shape[0]
    # input_dim is 4*50*50
    # output_size is 4 (up, down, left, right)
    def __init__(self, args, input_dim, output_size, device):
        super(MlpICMModel, self).__init__()
        # print("device is",device)
        self._use_feature_normalization = args.use_feature_normalization
        self._use_orthogonal = args.use_orthogonal
        self.hidden_size = args.hidden_size
        self.resnet_time = 4
        self.input_size = input_dim
        self.output_size = output_size
        # print("input size and output size", self.input_size, self.output_size)
        self.reward_scale = args.reward_scale
        self.policy_weight = args.policy_weight
        self.weight = args.weight
        self.device = device
        use_orthogonal = args.use_orthogonal
        # self.delta = args.
        # Initiate the encoder, forward model and inverse model with orthogonal method.
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def init_(m):
            return init_module(m, init_method, lambda x: nn.init.constant_(x, 0))
        # if self._use_feature_normalization:
        #     self.feature_norm = nn.LayerNorm(self.input_size).to(self.device)
        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(2500)
        # Encoder
        self.feature = nn.Sequential(
            init_(nn.Linear(self.input_size, self.hidden_size)),
            Swish(),
            init_(nn.Linear(self.hidden_size, self.hidden_size)),
            Swish(),
            init_(nn.Linear(self.hidden_size, self.hidden_size)),
        ).to(device)
        # self.feature = nn.Linear(self.input_size, self.hidden_size).to(device)
        #     # Swish(),
        #     # init_(nn.Linear(self.hidden_size, self.hidden_size)),
        #     # Swish(),
        #     # init_(nn.Linear(self.hidden_size, self.hidden_size)),


        # 输入的state是每个agent自己的whole map
        # *2 is because the input of inverse_net is both state and next_state,
        # while the input_dim is the dimension of state.
        self.inverse_net = nn.Sequential(
            init_(nn.Linear(self.hidden_size*2, self.hidden_size)),
            nn.LayerNorm(self.hidden_size),
            Swish(),
            init_(nn.Linear(self.hidden_size, self.hidden_size)),
            nn.LayerNorm(self.hidden_size),
            Swish(),
            # get the logits of actions
            init_(nn.Linear(self.hidden_size, self.output_size)),
        ).to(device)

        # self.residual = [nn.Sequential(init_(nn.Linear(self.output_size + self.hidden_size, self.hidden_size)),
        #     Swish(),
        #     init_(nn.Linear(self.hidden_size, self.hidden_size),
        # ).to(self.device))] * 2 * self.resnet_time

        self.forward_net_1 = nn.Sequential(
            init_(nn.Linear(self.output_size + self.hidden_size, self.hidden_size)),
            Swish(),
            init_(nn.Linear(self.hidden_size, self.hidden_size)),
            # Swish(),
            # nn.Linear(self.hidden_size, self.hidden_size),
            # Swish(),
            # nn.Linear(self.hidden_size, self.hidden_size),
            # Swish(),
            # nn.Linear(self.hidden_size, self.hidden_size)
        ).to(device)
        # self.forward_net_2 = nn.Sequential(
        #     init_(nn.Linear(self.output_size + 256, 256)),
        #     Swish(),
        #     nn.Linear(256, 256),
        #     Swish(),
        #     nn.Linear(256, 256),
        #     Swish(),
        #     nn.Linear(256, 256),
        #     Swish(),
        #     nn.Linear(256, 256)
        # )
        self.to(device)

    def forward(self, inputs):
        state, next_state, action = inputs
        state = torch.from_numpy(state).to(self.device)
        next_state = torch.from_numpy(next_state).to(self.device)

        # # 将 x 分解成 40, -1, 2500
        # batch = x.shape[0]
        # x_reshaped = x.view(batch, -1, 2500)
        # # 对最后一维应用 layer norm
        # x_normed = self.feature_norm(x_reshaped)
        # # 还原 x 的大小为 40, 7500
        # x = x_normed.view(x.size())
        #
        # x = self.mlp(x)


        if self._use_feature_normalization:
            batch = state.shape[0]
            size = state.size()
            state = self.feature_norm(state.view(batch, -1, 2500)).view(size)
            next_state = self.feature_norm(next_state.view(batch, -1, 2500)).view(size)

        # print("检查state的梯度信息", state.requires_grad)

        # Encode state and next_state
        # print("检查神经网络的梯度")
        # for param in self.feature.parameters():
            # print(param.requires_grad)

        encode_state = self.feature(state)
        encode_next_state = self.feature(next_state)
        # get predicted action. Then I need to compare the pred_action with the true action
        # By calculating the Norm2 distance, we can get the inverse loss.
        # print("encode_state.shape", encode_state.shape)
        pred_action = torch.cat((encode_state, encode_next_state), -1)
        # print("pred_action.shape", pred_action.shape)
        pred_action = self.inverse_net(pred_action)
        # print("pred_action  gard 2", pred_action.grad)
        # ---------------------

        # get pred next state and by comparing the predicted next state and the true next state,
        # we can get the curiosity loss.
        action_tensor = torch.from_numpy(action).to(self.device)
        pred_next_state_feature_orig = torch.cat((encode_state, action_tensor), -1).float()
        pred_next_state_feature_orig = self.forward_net_1(pred_next_state_feature_orig)

        # print("检查生成时的梯度")
        # print("pred_next_state_feature_orig", pred_next_state_feature_orig.requires_grad)
        # print("pred_action ", pred_action.requires_grad)
        return pred_next_state_feature_orig, encode_next_state, pred_action

    @torch.no_grad()
    # Just return the intrinsic reward
    def rewards(self, next_states_hat, next_states_latent):
        # return self.reward_scale / 2 * (next_states_hat - next_states_latent).norm(2, dim=tuple(range(1, next_states_hat.ndim))).pow(2)
        loss_func = F.smooth_l1_loss
        # print("type of next", type(next_states_latent))
        # print("next_states_hat[..., :]", next_states_hat[..., :].size())
        diff = loss_func(next_states_hat[..., -1], next_states_latent[..., -1], reduction='none')*self.hidden_size
        # Add a dimension of size 1 to get the reward tensor of shape [2000, 2, 1]
        # print("diff shape is", diff.shape)
        diff = diff.unsqueeze(-1)
        # print("diff shape", diff.shape)

        return self.reward_scale / 2 * diff.to('cpu')

    # Just return the intrinsic loss
    # Attention: both actions_hat and actions are the softmax


class CnnActorCriticNetwork(nn.Module):
    def __init__(self):
        super(CnnActorCriticNetwork, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, 512)

        self.actor = nn.Linear(512, 3)
        self.critic = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 7 * 7 * 64)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        policy = self.actor(x)
        value = self.critic(x)
        return policy, value




class ICMModel(nn.Module):
    def __init__(self, input_size, output_size, use_cuda=True):
        super(ICMModel, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        feature_output = 7 * 7 * 64
        self.feature = nn.Sequential(
            nn.Conv2d(
                in_channels=4,
                out_channels=32,
                kernel_size=8,
                stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(feature_output, 512)
        )

        self.inverse_net = nn.Sequential(
            nn.Linear(512 * 2, 512),
            nn.ReLU(),
            nn.Linear(512, output_size)
        )

        self.residual = [nn.Sequential(
            nn.Linear(output_size + 512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
        ).to(self.device)] * 8

        self.forward_net_1 = nn.Sequential(
            nn.Linear(output_size + 512, 512),
            nn.LeakyReLU()
        )
        self.forward_net_2 = nn.Sequential(
            nn.Linear(output_size + 512, 512),
        )

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.kaiming_uniform_(p.weight)
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.kaiming_uniform_(p.weight, a=1.0)
                p.bias.data.zero_()

    def forward(self, inputs):
        state, next_state, action = inputs

        encode_state = self.feature(state)
        encode_next_state = self.feature(next_state)
        # get pred action
        pred_action = torch.cat((encode_state, encode_next_state), 1)
        pred_action = self.inverse_net(pred_action)
        # ---------------------

        # get pred next state
        pred_next_state_feature_orig = torch.cat((encode_state, action), 1)
        pred_next_state_feature_orig = self.forward_net_1(pred_next_state_feature_orig)

        # residual
        for i in range(4):
            pred_next_state_feature = self.residual[i * 2](torch.cat((pred_next_state_feature_orig, action), 1))
            pred_next_state_feature_orig = self.residual[i * 2 + 1](
                torch.cat((pred_next_state_feature, action), 1)) + pred_next_state_feature_orig

        pred_next_state_feature = self.forward_net_2(torch.cat((pred_next_state_feature_orig, action), 1))

        real_next_state_feature = encode_next_state
        return real_next_state_feature, pred_next_state_feature, pred_action



