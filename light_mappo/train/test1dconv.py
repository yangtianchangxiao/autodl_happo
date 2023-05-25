import torch
import torch.nn as nn

# Define the 1D-CNN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc = nn.Linear(32*4, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 32*4)12
        x = self.fc(x)
        return x

# create an instance of the 1D-CNN
model = Net()

# input sequences of different lengths
input_sequence1 = torch.randn(1, 1, 10)
input_sequence2 = torch.randn(1, 1, 15)

# feed the input sequences to the 1D-CNN
output1 = model(input_sequence1)
output2 = model(input_sequence2)

print(output1.shape) # torch.Size([1, 2])
print(output2.shape) # torch.Size([1, 2])