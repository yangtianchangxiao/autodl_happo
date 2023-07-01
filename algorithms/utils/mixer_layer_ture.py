import torch.nn as nn
from .util import init, get_clones

"""MLP modules."""

    
class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, in_channels, embed_dim, stride_data):
        super().__init__()
        # 12 patch_size
        # stride 12
        # num_patch = (60/12)**2 = 25
        self.proj1 = nn.Unfold(kernel_size=patch_size, stride=stride_data)
        self.fc = nn.Linear(patch_size*patch_size*in_channels, embed_dim)

    def forward(self, x):
        # 输入形状: [batch_size, 1, 60, 60]
        x = self.proj1(x)  # 输出形状: [batch_size, patch_size*patch_size, num_patches]
        x = x.transpose(1, 2)  # 输出形状: [batch_size, num_patches, patch_size*patch_size]
        x = self.fc(x)  # 输出形状: [batch_size, num_patches, embed_dim]
        return x


#     def __init__(self, args, obs_shape, cat_self=True, attn_internal=False):

class MixerBase(nn.Module):
    def __init__(self, args, obs_shape):
        super().__init__()
        # Add these parameters to Config
        patch_size = args.patch_size
        in_channels = args.in_channels
        # embed_dim is acutally the embed_dim.
        embed_dim = args.embed_dim
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
        self.patch_embed = PatchEmbedding(patch_size, in_channels, embed_dim, stride)
        total_num_batch = (60//patch_size)**2
        # Althought I use GELU in fact but there is no GELU support in pytorch
        gain = nn.init.calculate_gain('relu')
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)
       
        if self._use_feature_normalization:
            # self.feature_norm = nn.LayerNorm(obs_dim)
            # self.token_norm = nn.LayerNorm(patch_size*patch_size*in_channels)
            # self.channel_norm = nn.LayerNorm(embed_dim)
            self.mixer_norm = nn.LayerNorm(embed_dim)
            # self.feature_norm = nn.LayerNorm(60*60)
            self.token_mlp = nn.Sequential(
                init_(nn.Linear(total_num_batch, total_num_batch)),
                nn.GELU(),
                nn.Dropout(p=self.dropout_prob),
                init_(nn.Linear(total_num_batch, total_num_batch)),

            )
            self.channel_mlp = nn.Sequential(
                init_(nn.Linear(embed_dim, embed_dim)),
                # self.channel_norm,
                nn.GELU(),
                nn.Dropout(p=self.dropout_prob),
                init_(nn.Linear(embed_dim, embed_dim)),
            )
        
        # self.fc_layers = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(num_layers)])
        self.mixer_layernorma = nn.LayerNorm(embed_dim*total_num_batch)
        self.fc_layers = nn.Linear(embed_dim, output_dim)

            
    def forward(self, x):
        self.batch = x.shape[0]
        original_size = x.size()
        # if self._use_feature_normalization:
        #     # print("x.shape",x.shape)
        #     # 将 x 分解成 40, -1, 2500
        #     x = x.view(self.batch, -1, 60*60)
        #     # 对最后一维应用 layer norm
        #     x = self.feature_norm(x)
        #     # 还原 x 的大小为 40, 7500
        #     x = x.view(original_size)
        x = x.view(self.batch, -1, 60, 60)  # Reshape to (batch_size, 4, 50, 50)
        x = self.patch_embed(x)  # Patch embedding
        for i in range(self.transpose_time):
            # 输出形状: [batch_size, num_patches, embed_dim]
            x = self.mixer_norm(x) # 对于每个通道进行归一化
            x = self.token_mlp(x.transpose(1, 2)).transpose(1, 2)  # 输出形状: [batch_size, num_patches, embed_dim]
            x = self.mixer_norm(x) # 对于每个通道进行归一化
            x = self.channel_mlp(x)  # 输出形状: [batch_size, num_patches, embed_dim]
        x = x.mean(dim=1)  # 输出形状: [batch_size, embed_dim]
        x = self.fc_layers(x)
        return x
