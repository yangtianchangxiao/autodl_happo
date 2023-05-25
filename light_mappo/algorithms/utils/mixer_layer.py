import torch.nn as nn
from .util import init, get_clones

"""MLP modules."""


class MixerLayer(nn.Module):
    def __init__(self, input_dim, hidden_size, layer_N, use_orthogonal, use_ReLU, use_Dropout, Dropout_prob):
        super(MixerLayer, self).__init__()
        self._layer_N = layer_N

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.token_mlp = init_(nn.Linear(120, 120))
        self.channel_mlp = init_(nn.Linear(120, 120))
        if use_Dropout:
            # self.fc1 = nn.Sequential(
            #     init_(nn.Linear(input_dim, hidden_size)), active_func, nn.Linear(hidden_size),
            #     nn.Dropout(p=Dropout_prob))
            # self.fc_h = nn.Sequential(init_(
            #     nn.Linear(hidden_size, hidden_size)), active_func, nn.LayerNorm(hidden_size),
            #     nn.Dropout(p=Dropout_prob))
            self.fc1 = nn.Sequential(
                init_(nn.Linear(input_dim, hidden_size)), active_func, nn.LayerNorm(hidden_size),
                nn.Dropout(p=Dropout_prob))
            self.fc_h = nn.Sequential(init_(
                nn.Linear(hidden_size, hidden_size)), active_func, nn.LayerNorm(hidden_size),
                nn.Dropout(p=Dropout_prob))
        else:
            self.fc1 = nn.Sequential(
                init_(nn.Linear(input_dim, hidden_size)), active_func, nn.LayerNorm(hidden_size))
            self.fc_h = nn.Sequential(init_(
                nn.Linear(hidden_size, hidden_size)), active_func, nn.LayerNorm(hidden_size))
        self.fc2 = get_clones(self.fc_h, self._layer_N)

    def forward(self, x, batch):
        # print("start")
        # 调整 x 的形状以应用 token-mixing 和 channel-mixing
        x = x.view(batch, -1, 120, 120)  # Reshape to (batch_size, 4, 50, 50)

        # Token-mixing
        x = self.token_mlp(x)

        # Channel-mixing
        x = x.permute(0, 2, 1, 3)  # Swap dimensions to (batch_size, 50, 4, 50)
        x = self.channel_mlp(x)
        x = x.permute(0, 2, 1, 3)  # Swap dimensions back to (batch_size, 4, 50, 50)
        x = x.reshape(batch, -1)
        x = self.fc1(x)
        # print("x")
        for i in range(self._layer_N):
            x = self.fc2[i](x)
            # a = [0]
            # if x.requires_grad is True:
            #     a >7
        return x


class MixerBase(nn.Module):
    def __init__(self, args, obs_shape, cat_self=True, attn_internal=False):
        super(MixerBase, self).__init__()

        self._use_feature_normalization = args.use_feature_normalization
        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        self._stacked_frames = args.stacked_frames
        self._layer_N = args.layer_N
        self.hidden_size = args.hidden_size
        self._add_dropout = args.add_dropout
        self.dropout_prob = args.dropout_prob

        obs_dim = obs_shape[0]
        # print("obs_dim is!!!!",obs_dim)
        if self._use_feature_normalization:
            # self.feature_norm = nn.LayerNorm(obs_dim)
            self.feature_norm = nn.LayerNorm(120*120)
            # self.feature_norm = nn.BatchNorm1d(obs_dim)
            # self.feature_norm = nn.GroupNorm(4,4)
        self.mlp = MixerLayer(obs_dim, self.hidden_size,
                              self._layer_N, self._use_orthogonal, self._use_ReLU, self._add_dropout, self.dropout_prob)

    def forward(self, x):
        batch = x.shape[0]
        original_size = x.size()
        if self._use_feature_normalization:
            # print("x.shape",x.shape)

            # 将 x 分解成 40, -1, 2500

            x_reshaped = x.view(batch, -1, 120*120)
            # 对最后一维应用 layer norm
            x_normed = self.feature_norm(x_reshaped)
            # 还原 x 的大小为 40, 7500

            x = x_normed.view(original_size)

            # x = self.feature_norm(x)
            x = self.mlp(x, batch)
        else:
            x = self.mlp(x, batch)
        return x