
import pandas as pd
import json
import nltk
from gensim.models import Word2Vec
from termcolor import colored


# load data frame
def load_dataframe(file: str=""):
    return pd.read_csv(file, low_memory=False)

# load json content
def load_json(file: str="") -> dict:
    with open(file, "r") as f:
        return json.load(f)

# load json content
def write_json(content: list, file: str=""):
    with open(file, "w+") as f:
        return json.dump(content, f)

# get the frequency of one word in dataset
def get_word_frequency(preprocessed_dataset: list, word: str=""):
    total_words = []
    for tweet in preprocessed_dataset: total_words.extend(tweet)

    return nltk.FreqDist(total_words)[word]

# unzip zipped lists
def unzip(zipped: list) -> list:
    list1, list2 = [list(s) for s in zip(*zipped)]
    return list1, list2

# load Word2Vec model
def load_gensim_model(file: str=""):
    return Word2Vec.load(file)

# print trainings progress
def print_progress(epochs: int, epoch: int, loss: float, val_acc: float):
    epochs = colored(epoch+1, "cyan", attrs=['bold']) + colored("/", "cyan", attrs=['bold']) + colored(epochs, "cyan", attrs=['bold'])
    loss = colored(round(loss, 6), "cyan", attrs=['bold'])
    val_acc = colored(round(val_acc, 4), "cyan", attrs=['bold']) + colored("%", "cyan", attrs=['bold'])

    print(" ")
    print("\nepoch {} - loss: {} - val_acc: {}".format(epochs, loss, val_acc))
    print("\n..........................................................................................\n")