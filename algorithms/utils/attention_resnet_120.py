import torch
from torch import nn
import torch.nn.functional as F

class resnet(nn.Module):
    def __init__(self, dropout_prob):
        super(resnet, self).__init__()
        # 将60*60 缩小一半到 30*30
        self.conv_half1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride=2, padding=1)
        # self.batch_norm32 = nn.BatchNorm2d(32)
        # self.batch_norm64 = nn.BatchNorm2d(64)
        self.batch_norm32_15_15 = nn.LayerNorm([32, 15, 15])
        self.batch_norm32_30_30 = nn.LayerNorm([32, 30, 30])
        self.batch_norm64_30_30 = nn.LayerNorm([64, 30, 30])
        self.batch_norm64_15_15 = nn.LayerNorm([64, 15, 15])
        
        self.activate = nn.ELU()
        self.conv_half2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.conv_half3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1)
        
        self.conv_same1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv_same2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        
        self.init_sequen = nn.Sequential(
            self.conv_half1,
            # self.batch_norm32,
            self.batch_norm32_30_30,
            self.activate,
            nn.Dropout(0)  # 添加dropout，丢弃率为0.5
        )

        self.residual = nn.Sequential(
            self.conv_half2,
            self.batch_norm64_15_15,
            self.activate,
            nn.Dropout(0),  # 添加dropout，丢弃率为0.5
            self.conv_same1,
            # self.batch_norm64,
            self.batch_norm64_15_15,
            self.activate,
            nn.Dropout(0),  # 添加dropout，丢弃率为0.5
            self.conv_same2,
            # self.batch_norm64,
            self.batch_norm32_15_15,
            self.activate,
            nn.Dropout(0)   # 添加dropout，丢弃率为0.5
        )
        self.conv_local_view = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=1, stride=2, padding=0) # input 9*9; output （9-2+2*0）/2+1=5
        self.bath_norm_local_view = nn.LayerNorm([32, 5, 5])
        self.local_conv = nn.Sequential(
            self.conv_local_view,
            self.bath_norm_local_view,
            self.activate
        )
    def forward(self, x, batch_size):
        x1 = x[:, :, :60*60].view(batch_size, -1, 60, 60)
        x2 = x[:, :, 60*60:].view(batch_size, -1, 9, 9)
        # print("x1 shape is", x1.shape)
        # print("x2 shape is", x2.shape)
        x = residual_x = self.init_sequen(x1) # batch_size, 4, 30, 30
        # test = self.conv_half2(x)
        # print("Shape of test:", test.shape) # Shape of test: torch.Size([32, 8, 15, 15])
        # test2 = self.conv_same1(test)
        # print("Shape of test2:", test2.shape) # Shape of test2: torch.Size([32, 8, 15, 15])
        # test3 = self.conv_same2(test2)
        # print("Shape of test3:", test3.shape) # Shape of test3: torch.Size([32, 4, 15, 15])
        residual_x  = self.residual(residual_x) 
        residual_x = F.interpolate(input=residual_x, scale_factor=2, mode='bilinear', align_corners=True)  # batch_size, 4, 30, 30
        # print("Shape of x:", x.shape) # Shape of x: torch.Size([32, 4, 31, 31])                                                                                 
# Shape of residual_x: torch.Size([32, 4, 62, 62])
        # print("Shape of residual_x:", residual_x.shape)
        # x = torch.cat((x, residual_x), dim=1) # batch_size, 8, 30, 30
        x = x + residual_x # 
        x = self.conv_half3 (x) # batch_size, 8, 15, 15
        x = self.batch_norm32_15_15(x)
        x = self.activate(x) # output batch_size, 32, 15, 15
        x2 = self.local_conv(x2) # output batch_size, 32, 5, 5
        return x, x2
        
class Attention_model(nn.Module):
    def __init__(self, args, obs_shape):
        super(Attention_model, self).__init__()
        # self.proj1 = nn.Unfold(kernel_size=patch_size, stride=stride_data)
        d_model=args.d_model
        nhead=args.nhead
        hidden_dim = args.hidden_size
        dropout_prob = args.dropout_prob

        self.resnet = resnet(dropout_prob)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead)
        # self.mlp = nn.Sequential(
        #     nn.Linear(d_model, hidden_dim),  
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, output_dim)  
        # )
        mlp_input = int(d_model*(15*15+5*5)) # I have half size of map twice
        self.mlp = nn.Linear(mlp_input, hidden_dim)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_prob)
        self.state_length = 60*60+9*9 # 9*9 is local_view_range
    def forward(self, x):
        # x = self.proj1(x)  # shape: [batch_size, patch_size*patch_size, num_patches] = [4, 100, 36]
        batch_size = x.size(0)  
        x = x.view(batch_size, -1, self.state_length)  # Reshape to (batch_size, 4, 50, 50)
        x, x2 = self.resnet(x, batch_size) # x shape: [batch_size, 32, 15, 15]; x2 shape: [batch_size, 32, 5, 5]
        
        # Flatten spatial dimensions and transpose for multi-head attention input
        x = torch.cat((x.view(x.size(0), x.size(1), -1), x2.view(x2.size(0), x2.size(1), -1)), axis=-1)
        x = x.permute(2,0,1)  # output shape: seq_len (15*15+5*5), batch_size, d_model 
        x, _ = self.multihead_attn(x, x, x)  # seq_len (15*15+5*5), batch_size, d_model ()
        x = x.permute(1, 0, 2).contiguous().view(batch_size, -1)  # batch_size, seq_len * d_model
    
        # Option 1: Average Pooling + Linear
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        # Option 2: MLP
        x = self.dropout(x)
        x = self.mlp(x)
        
        return x


