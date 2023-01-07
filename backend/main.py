from fastapi import FastAPI
from typing import List
# from pydantic import BaseModel
import pandas as pd
import numpy as np

import torch
from model.cnn import CNN

app = FastAPI()

# load model
model = CNN()
model.load_state_dict(torch.load('./model/cnn.pt', map_location=torch.device('cpu')))



@app.post("/predict_cnn")
async def predict_cnn(data: List):
    ar_data = np.array(data)
    # newaxis
    ar_data = ar_data[np.newaxis, np.newaxis, :, :]
    # numpy
    ts = torch.from_numpy(ar_data.astype('float32'))
    # model
    output = model(ts)[0]
    ar_output = pd.Series(output.detach().numpy())
    ar_softmax_output = np.exp(ar_output)/np.sum(np.exp(ar_output))
    dict_output = ar_softmax_output.to_dict()
    return dict_output