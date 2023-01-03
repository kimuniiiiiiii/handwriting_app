# Description
畳み込みニューラルネットワークによる手書き数字認識アプリです。

学習データは28×28(＝784ピクセル)の60,000サンプルの手書き数字データベースである[MNIST](http://yann.lecun.com/exdb/mnist/)を利用。

↓をクリックしてアプリ利用可能です。

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://kimuniiiiiiii-handwriting-app-app-z507ha.streamlit.app/)

----------------------------------------------------------------
## Model

    　　Layer (type)         Output Shape   　  # of Params
        Conv2d-1         [-1, 16, 28, 28]             160
        ReLU-2           [-1, 16, 28, 28]               0
        MaxPool2d-3      [-1, 16, 14, 14]               0
        Conv2d-4         [-1, 32, 14, 14]           4,640
        ReLU-5           [-1, 32, 14, 14]               0
        MaxPool2d-6      [-1, 32, 7, 7]                 0
        Linear-7         [-1, 1024]             1,606,656
        ReLU-8           [-1, 1024]                     0
        Linear-9         [-1, 1024]             1,049,600
        ReLU-10          [-1, 1024]                     0
        Linear-11        [-1, 10]                  10,250

----------------------------------------------------------------

## Model Code(pytorch)
```python
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
        h = self.pool(self.act(self.conv1(x)))
        h = self.pool(self.act(self.conv2(h)))
        h = h.view(h.size()[0], -1)
        h = self.act(self.l1(h))
        h = self.act(self.l2(h))
        h = self.l3(h)
        return h

```