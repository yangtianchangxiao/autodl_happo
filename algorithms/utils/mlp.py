import torch.nn as nn
from .util import init, get_clones

"""MLP modules."""

class MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_size, layer_N, use_orthogonal, use_ReLU, use_Dropout=False, Dropout_prob=0.2):
        super(MLPLayer, self).__init__()
        self._layer_N = layer_N

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        if use_Dropout:
            # self.fc1 = nn.Sequential(
            #     init_(nn.Linear(input_dim, hidden_size)), active_func, nn.Linear(hidden_size),
            #     nn.Dropout(p=Dropout_prob))
            # self.fc_h = nn.Sequential(init_(
            #     nn.Linear(hidden_size, hidden_size)), active_func, nn.LayerNorm(hidden_size),
            #     nn.Dropout(p=Dropout_prob))
            self.fc1 = nn.Sequential(
                init_(nn.Linear(input_dim, hidden_size)), active_func, nn.LayerNorm(hidden_size), nn.Dropout(p=Dropout_prob))
            self.fc_h = nn.Sequential(init_(
                nn.Linear(hidden_size, hidden_size)), active_func, nn.LayerNorm(hidden_size), nn.Dropout(p=Dropout_prob))
        else:
            self.fc1 = nn.Sequential(
                init_(nn.Linear(input_dim, hidden_size)), active_func, nn.LayerNorm(hidden_size))
            self.fc_h = nn.Sequential(init_(
                nn.Linear(hidden_size, hidden_size)), active_func, nn.LayerNorm(hidden_size))
        self.fc2 = get_clones(self.fc_h, self._layer_N)

    def forward(self, x):
        x = self.fc1(x)
        for i in range(self._layer_N):
            x = self.fc2[i](x)
            # a = [0]
            # if x.requires_grad is True:
            #     a >7
        # print("x.shape", x.shape)
        return x


class MLPBase(nn.Module):
    def __init__(self, args, obs_shape, cat_self=True, attn_internal=False):
        super(MLPBase, self).__init__()

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
            self.feature_norm = nn.LayerNorm(60*60)
            # self.feature_norm = nn.BatchNorm1d(obs_dim)
            # self.feature_norm = nn.GroupNorm(4,4)
        self.mlp = MLPLayer(obs_dim, self.hidden_size,
                              self._layer_N, self._use_orthogonal, self._use_ReLU, self._add_dropout, self.dropout_prob)

    def forward(self, x):
        if self._use_feature_normalization:
            # print("x.shape",x.shape)

            # 将 x 分解成 40, -1, 2500
            batch = x.shape[0]
            x_reshaped = x.view(batch, -1, 60*60)
            # 对最后一维应用 layer norm
            x_normed = self.feature_norm(x_reshaped)
            # 还原 x 的大小为 40, 7500
            x = x_normed.view(x.size())

            # x = self.feature_norm(x)
            x = self.mlp(x)
        else:
            x = self.mlp(x)
        return x