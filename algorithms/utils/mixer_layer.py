import torch.nn as nn
from .util import init, get_clones

"""MLP modules."""


class MixerLayer(nn.Module):
    def __init__(self, input_dim, hidden_size, layer_N, use_orthogonal, use_ReLU, use_Dropout, Dropout_prob):
        super(MixerLayer, self).__init__()
        self._layer_N = layer_N

        active_func = [nn.Tanh(), nn.ReLU(inplace=True)][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.token_mlp = init_(nn.Linear(60, 60))
        self.channel_mlp = init_(nn.Linear(60, 60))
        if use_Dropout:
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
        x = x.view(batch, -1, 60, 60)  # Reshape to (batch_size, 4, 50, 50)
        # Token-mixing
        x = self.token_mlp(x)
        # Channel-mixing
        x = x.permute(0, 2, 1, 3)  # Swap dimensions to (batch_size, 60, 4, 60)
        x = self.channel_mlp(x)
        x = x.permute(0, 2, 1, 3)  # Swap dimensions back to (batch_size, 4, 60, 60)
        x = x.reshape(batch, -1)
        x = self.fc1(x)
        for i in range(self._layer_N):
            x = self.fc2[i](x)
        return x


class MixerBase2(nn.Module):
    def __init__(self, args, obs_shape, cat_self=True, attn_internal=False):
        super().__init__()

        self._use_feature_normalization = args.use_feature_normalization
        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        self._stacked_frames = args.stacked_frames
        self._layer_N = args.layer_N
        self.hidden_size = args.hidden_size
        self._add_dropout = args.add_dropout
        self.dropout_prob = args.dropout_prob

        obs_dim = obs_shape[0]
        # print("self feature normalization is", self._use_feature_normalization)
        if self._use_feature_normalization:
            # self.feature_norm = nn.LayerNorm(obs_dim)
            self.feature_norm = nn.LayerNorm(60*60)
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
            x_reshaped = x.view(batch, -1, 60*60)
            # 对最后一维应用 layer norm
            x_normed = self.feature_norm(x_reshaped)
            # 还原 x 的大小为 40, 7500
            x = x_normed.view(original_size)
            x = self.mlp(x, batch)
        else:
            x = self.mlp(x, batch)
        return x
    
    
class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, in_channels, embed_dim, stride_data):
        super().__init__()
        # 默认是 3*3 的卷积核，strid=1
        # (60-3)/1 = 58
        # self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=stride_data)
        # self.proj3 = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        # self.proj4 = nn.Conv2d(4, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.proj1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1)
        self.proj8 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=5, stride=2)

        self.activate = nn.LeakyReLU(inplace= True)
        # # 58/2 = 29
        self.maxpool2 = nn.AvgPool2d(kernel_size=2, stride=2)  
        self.maxpool3 = nn.AvgPool2d(kernel_size=3, stride=2)   
    def forward(self, x):
        # if x.shape[1] == 3:
        #     x = self.proj3(x)  # (batch_size, embed_dim, num_patches^(1/2), num_patches^(1/2))
        # else:
        #     x = self.proj4(x)
        # 30
        # (60-3)/1+1 = 58
        x = self.proj1(x)
        x = self.activate(x)
        x = self.maxpool2(x) # 29，29
        # print("x.shape", x.shape)
        x = self.proj8(x) # (29-5)/2 + 1= 13
        x = self.activate(x)
        x = self.maxpool3(x) #(13-3)/2 + 1  = 6
        # print("x.shape2 is", x.shape)
        x = x.flatten(2)
        # print("x.shape3 is", x.shape)
        # (batch_size, embed_dim, num_patches)
        # print("x.shape",x.shape)
        # # x = x.transpose(1, 2)  # (batch_size, num_patches, embed_dim)
        # print(x.shape)
        return x


#     def __init__(self, args, obs_shape, cat_self=True, attn_internal=False):

class MixerBase(nn.Module):
    def __init__(self, args, obs_shape):
        super().__init__()
        # Add these parameters to Config
        patch_size = args.patch_size
        in_channels = args.in_channels
        # Hidden_dim is acutally the embed_dim.
        hidden_dim = args.hidden_dim
        stride = args.stride
    
        # This is hidden_size is the size of final selet network
        output_dim = args.hidden_size
        
        self.transpose_time = args.transpose_time
        
        self._use_feature_normalization = args.use_feature_normalization
        use_orthogonal = args.use_orthogonal
        self._stacked_frames = args.stacked_frames
        self._layer_N = args.layer_N
        self._add_dropout = args.add_dropout
        self.dropout_prob = args.dropout_prob
        # active_func = nn.GELU(inplace=True)
        
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        
        self.patch_embed = PatchEmbedding(patch_size, in_channels, hidden_dim, stride)
        # total_num_batch = int((60/patch_size)**2) # If 60 could / patch_size totally
        total_num_batch = 36 
        hidden_dim =32
        # Althought I use GELU in fact but there is no GELU support in pytorch
        gain = nn.init.calculate_gain('relu')
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)
       
        if self._use_feature_normalization:
            # self.feature_norm = nn.LayerNorm(obs_dim)
            self.token_norm = nn.LayerNorm(total_num_batch)
            self.channel_norm = nn.LayerNorm(hidden_dim)
            self.feature_norm = nn.LayerNorm(60*60)
            self.token_mlp = nn.Sequential(
                init_(nn.Linear(total_num_batch, total_num_batch)),
                self.token_norm,
                nn.ReLU(inplace= True),
                nn.Dropout(p=self.dropout_prob)
            )
            self.channel_mlp = nn.Sequential(
                init_(nn.Linear(hidden_dim, hidden_dim)),
                self.channel_norm,
                nn.ReLU(inplace= True),
                nn.Dropout(p=self.dropout_prob)
            )
        else:
            self.token_mlp = nn.Sequential(
                init_(nn.Linear(total_num_batch, total_num_batch)),
                nn.GELU(),
                nn.Dropout(p=self.dropout_prob)
            )
            self.channel_mlp = nn.Sequential(
                init_(nn.Linear(hidden_dim, hidden_dim)),
                nn.GELU(),
                nn.Dropout(p=self.dropout_prob)
            )
        # self.fc_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.fc_layers = nn.Linear(hidden_dim*total_num_batch, output_dim)

            
    def forward(self, x):
        self.batch = x.shape[0]
        original_size = x.size()
        if self._use_feature_normalization:
            # print("x.shape",x.shape)
            # 将 x 分解成 40, -1, 2500
            x = x.view(self.batch, -1, 60*60)
            # 对最后一维应用 layer norm
            x = self.feature_norm(x)
            # 还原 x 的大小为 40, 7500
            x = x.view(original_size)
        x = x.view(self.batch, -1, 60, 60)  # Reshape to (batch_size, 4, 50, 50)
        
        x = self.patch_embed(x)  # Patch embedding
        for i in range(self.transpose_time):
            x = self.token_mlp(x)  # Token mixing
            # print("111111")
            x = x.transpose(1, 2)  # Prepare for channel mixing
            x = self.channel_mlp(x)  # Channel mixing
            x = x.transpose(1, 2)  # Prepare for FC layers
        x = x.reshape(self.batch, -1)
        x = self.fc_layers(x)
        return x
