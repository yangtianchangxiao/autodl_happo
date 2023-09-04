
# 对不起，我之前的回答中的计算有误。你是对的，ResNet-18的第一层卷积核是7x7的。我来纠正一下：

# 假设输入图像的形状为（batch_size, 3, 60, 60），经过ResNet-18的每一层后，你会得到以下形状的输出：

# 1. **Convolution Layer 1**：（batch_size, 64, 30, 30）。这是因为第一层是一个7x7的卷积核，步长为2，填充为3，所以输出的大小是(60+2*3-7)/2+1=30，通道数变为64。

# 2. **Max Pooling Layer**：（batch_size, 64, 15, 15）。这是因为最大池化层的核大小为3x3，步长为2，所以输出的大小是(30-3)/2+1=15，通道数保持为64。

# 3. **Residual Block 1 (2 layers)**：（batch_size, 64, 15, 15）。在这个残差块中，所有的卷积层的步长都是1，所以输出的大小不变，仍然是15x15，通道数保持为64。

# 4. **Residual Block 2 (2 layers)**：（batch_size, 128, 8, 8）。在这个残差块的第一层，卷积核的步长为2，所以输出的大小是(15-1)/2+1=8，通道数变为128。

# 5. **Residual Block 3 (2 layers)**：（batch_size, 256, 4, 4）。同样，在这个残差块的第一层，卷积核的步长为2，所以输出的大小是(8-1)/2+1=4，通道数变为256。

# 6. **Residual Block 4 (2 layers)**：（batch_size, 512, 2, 2）。在这个残差块的第一层，卷积核的步长为2，所以输出的大小是(4-1)/2+1=2，通道数变为512。

# 7. **Average Pooling Layer**：（batch_size, 512, 1, 1）。这是因为平均池化层的核大小为2x2，所以输出的大小是2/2=1，通道数保持为512。

# 8. **Fully Connected Layer**：（batch_size, 1000）。全连接层将512个通道的1x1图像转换为1000个输出，这通常对应于分类任务的类别数。

# 这就是如何计算ResNet-18每一层的输出形状。再次对之前的错误表示歉意。



import torch.optim as optim
import torch.nn as nn
import torch
import torchvision.models as models
from .util import init, get_clones

class Resnet18(nn.Module):
    def __init__(self, args, obs_shape):
        super(Resnet18, self).__init__()
        use_orthogonal = args.use_orthogonal
        output_dim = args.hidden_size
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain('relu')
        lr_resnet = args.lr_resnet
        lr_others = args.lr_others
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)
        # Load the pretrained model
        self.base_model = models.resnet18(pretrained=True)  # Set pretrained=False as we are going to modify the input layer

        # Modify the first convolution layer to accept single-channel input
        self.base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Remove the last fully connected layer
        modules = list(self.base_model.children())[:-1]  # all layers except the last one

        # Add a new fully connected layer
        self.fc = init_(nn.Linear(512, output_dim))

        # Set different learning rates for the base model and the fully connected layer
     

        self.base_model = nn.Sequential(*modules)
        
    def forward(self, x):
        self.batch = x.shape[0]
        x = x.view(self.batch, -1, 60, 60)  # Reshape to (batch_size, 4, 50, 50)
        
        x = self.base_model(x)  
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        return x
