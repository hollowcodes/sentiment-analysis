
from utils import load_gensim_model, load_json, print_progress
from model import Model

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.optim import lr_scheduler


class Run:
    def __init__(self, embeddings_model_path: str="", train_set_path: str="", test_set_path: str="", val_set_path: str="",
                 epochs: int=10, lr: float=0.001, batch_size: int=32):

        self.embeddings_model = load_gensim_model(embeddings_model_path)
        self.train_set = load_json(train_set_path)
        self.test_set = load_json(test_set_path)
        self.val_set = load_json(val_set_path)
        
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size

        self.model = Model().cuda()

    """ convert words in sentence to vector embeddings and convert to tensor """
    def _preprare_sample(self, sample: list):
        tokenized_sentence, target = sample[0], sample[1]
        embedded_sentence = [torch.Tensor(np.array(self.embeddings_model[word]).reshape(1, 32)) for word in tokenized_sentence]
        embedded_sentence = torch.cat(embedded_sentence).view(len(embedded_sentence), 1, -1).cuda()

        target = torch.Tensor([target]).cuda()

        return embedded_sentence, target

    """ evaluate accuracy of the model using validation data """
    def _evaluate(self, dataset) -> float:
        total = len(dataset)
        correct = 0
        for sample in self.val_set:
            embeddings, target = self._preprare_sample(sample)

            prediction = self.model.eval()(embeddings)

            is_correct = prediction.round().eq(target).detach().cpu().numpy()
            if is_correct:
                correct += 1

        return total / correct

    """ train model """
    def train(self):
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)

        loss_history = []
        val_acc_history = []
        for epoch in range(self.epochs):

            epoch_loss = []
            for sample in tqdm(self.train_set):
                optimizer.zero_grad()

                embeddings, target = self._preprare_sample(sample)
                prediction = self.model.train()(embeddings)

                loss = criterion(prediction.squeeze(-1), target)
                loss.backward()
                optimizer.step()

                epoch_loss.append(loss.item())

            current_loss = round(np.mean(epoch_loss), 7)
            current_val_acc = round(self._evaluate(self.val_set), 7)
            loss_history.append(current_loss)
            val_acc_history.append(current_val_acc)

            print_progress(self.epochs, epoch, current_loss, current_val_acc)

            if epoch % 5 == 0:
                torch.save(self.model.state_dict(), "models/model_1.pt")

    """ test model """
    def test(self):
        accuracy = self._evaluate(self.train_set)
        print("test-accracy: " + str(round(accuracy, 7)) + "%\n")

        
run = Run(embeddings_model_path="dataset/data/word_embeddings.model",
          train_set_path="dataset/data/train_set_tweets.json",
          test_set_path="dataset/data/test_set_tweets.json",
          val_set_path="dataset/data/val_set_tweets.json",
          epochs=50,
          lr=0.001,
          batch_size=32)

run.train()
run.test()
