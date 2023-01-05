import streamlit as st
from streamlit_drawable_canvas import st_canvas
import seaborn as sns
import torch

from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.cnn import CNN

# load model
model = CNN()
model.load_state_dict(torch.load('model/cnn.pt', map_location=torch.device('cpu')))

drawing_mode = 'freedraw'
stroke_width = 20
stroke_color = "#000000"
bg_color = "#eee"
realtime_update = True
width = 28
height = 28
max_estimates = -1


def write():

    # Tab
    tab1, tab2 = st.tabs(["App", "Overview"])

    with tab1:
        canvas()
    with tab2:
        st.write(md)


def canvas():
    st.title('手書き数字認識 / Handwriting Recognition')
    """### 下のキャンバスに数字を描いてください！"""

    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        update_streamlit=realtime_update,
        drawing_mode=drawing_mode,
        point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
        display_toolbar='red', #st.sidebar.checkbox("Display toolbar", True),
        # key="handwriting_recognition_app",
        )
    st.write('    ↑このボタンでキャンバスを白紙にできます。')

    if canvas_result.image_data is not None:

        img = Image.fromarray(canvas_result.image_data)
        img = img.convert(mode="L")
        img = ImageOps.invert(img)
        img_resize_lanczos = img.resize((width, height), Image.Resampling.LANCZOS)

        ar_resize = np.array(img_resize_lanczos)
        ar_resize = ar_resize - ar_resize.min()
        ar_resize = ar_resize/ar_resize.max()
        ar_resize = ar_resize[np.newaxis, np.newaxis, :, :]
        ts_resize = torch.from_numpy(ar_resize.astype('float32'))

        # バッチサイズ1なので、1つ目の出力だけ取り出す
        output = model(ts_resize)[0]
        s_output = pd.Series(output.detach().numpy())
        s_softmax_output = np.exp(s_output)/np.sum(np.exp(s_output))


        # 最も確率の高い数字を抽出
        max_estimates = np.argmax(s_softmax_output)
        if max_estimates>=0:
            st.header(f'あなたの書いた数字は「{max_estimates}」ですか？')

            col1, col2 = st.columns(2)
            with col1:
                fig = plt.figure()
                sns.heatmap(pd.DataFrame(ar_resize[0, 0, :, :]))
                st.write(fig)
                st.caption('28×28に変換')
            with col2:
                st.bar_chart(s_softmax_output.rename('各数字である確率'))
        else:
            pass        
        
md = """
# 所感
- 正解率は筆跡に依存すると思われるが、自分の手書きでは6, 8, 9が不正解となりやすい。MNISTにやや過学習しているか。正則化や平均化など試すとアウトオブサンプルなテストデータの読み取り精度向上出来るのではないか。
- MNISTは28×28と低解像度のため、モデルは低解像度前提のモデルとなっている。人が認識可能な程度で少し解像度の高い学習データを用いると、人の直感に整合的な結果が得られるのではないか。

# 推論モデルについて

###### pytorchでConvolutional Neural Networkによるモデル構築を行い、MNIST（28×28の手書き数字画像60,000枚）で学習を行いました。
----------------------------------------------------------------
### Learning
- optimizer : SGD (learning rate=0.01, momentum=0.9)
- loss function     : Cross entropy loss
- epoch     : 20
- batch size: 100

----------------------------------------------------------------
### Model

        Layer (type)               Output Shape      # of Params
            Conv2d-1           [-1, 16, 28, 28]             160
            ReLU-2           [-1, 16, 28, 28]               0
        MaxPool2d-3           [-1, 16, 14, 14]               0
            Conv2d-4           [-1, 32, 14, 14]           4,640
            ReLU-5           [-1, 32, 14, 14]               0
        MaxPool2d-6             [-1, 32, 7, 7]               0
            Linear-7                 [-1, 1024]       1,606,656
            ReLU-8                 [-1, 1024]               0
            Linear-9                 [-1, 1024]       1,049,600
            ReLU-10                 [-1, 1024]               0
        Linear-11                   [-1, 10]          10,250

----------------------------------------------------------------
### Model Code(pytorch)
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
"""