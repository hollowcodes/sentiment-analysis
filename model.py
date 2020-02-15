
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import load_json, load_gensim_model
import numpy as np


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.hidden = (torch.randn(1, 1, 32).cuda(), torch.randn(1, 1, 32).cuda())

        self.lstm = nn.LSTM(32, 32)
        self.linear = nn.Linear(32, 1)

    def forward(self, x):
        out, hidden = self.lstm(x, self.hidden)
        x = self.linear(out)
        x = torch.sigmoid(x)

        return x[-1]


"""gensim_model = load_gensim_model("dataset/data/word_embeddings.model")

sentence = [torch.Tensor(np.array(gensim_model[word]).reshape(1, 32)) for word in load_json("dataset/data/train_set_tweets.json")[0][0]]
sentence = torch.cat(sentence).view(len(sentence), 1, -1)

model = Model().train()
output = model(sentence)
print(output)"""

