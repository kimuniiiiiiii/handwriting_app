import torch
import torch.nn as nn
import torch.nn.functional as f

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.l1 = nn.Linear(7*7*32, 1024)
        self.l2 = nn.Linear(1024, 1024)
        self.l3 = nn.Linear(1024, 10)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):                       
        # 入力画像Shape: [batch, 1, 28, 28]
        h = self.pool(self.act(self.conv1(x)))  # [batch, 16, 14, 14]
        h = self.pool(self.act(self.conv2(h)))  # [batch, 32, 7, 7]
        h = h.view(h.size()[0], -1) # 配列サイズを変更
        h = self.act(self.l1(h))
        h = self.act(self.l2(h))
        h = self.l3(h)
        return h