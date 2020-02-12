
from utils import load_dataframe, write_json, unzip, get_word_frequency, load_json, load_gensim_model
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from tqdm import tqdm
from gensim.models import Word2Vec, KeyedVectors
import numpy as np
import os


class Preprocess:
    def __init__(self, dataframe, batch_name: str=""):
        self.batch_name = batch_name

        self.dataframe = dataframe
        self.tweets = self.dataframe["tweet"].tolist()

        # no labels when preprocessing test-set
        if self.batch_name != "test-set":
            self.labels = self.dataframe["label"].tolist()

        self.length = len(self.tweets)

        self.tokenizer = RegexpTokenizer(r"\w+")
        self.stop_words = set(stopwords.words("english"))
        self.ps = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

    # convert sentence to list
    def _tokenize(self, tweet: str) -> list:
        return self.tokenizer.tokenize(tweet)

    # remove stop-words
    def _rm_stop_words(self, tweet: list) -> list:
        return [word.lower() for word in tweet if not word in self.stop_words]

    # stem words
    def _stem(self, tweet: list) -> list:
        return [self.ps.stem(word) for word in tweet]

    # lemmetize nouns
    def _lemmatize(self, tweet: list) -> list:
        return [self.lemmatizer.lemmatize(word) for word in tweet]

    # apply nltk preprocessing steps on one tweet
    def _apply(self, tweet: str):
        tweet = self._tokenize(tweet)
        tweet = self._rm_stop_words(tweet)
        tweet = self._stem(tweet)
        tweet = self._lemmatize(tweet)
        return tweet

    # preprocess all tweets
    def preprocess_dataset(self):
        for idx in tqdm(range(len(self.tweets)), ncols=100, desc=("preprocessing " + self.batch_name)):
            self.tweets[idx] = self._apply(self.tweets[idx])

        # no labels when preprocessing test-set
        if self.batch_name != "test-set":
            return list(zip(self.tweets, self.labels))
        else:
            return self.tweets


""" load, preprocess and save train-/test-/val-set """
def create_preprocessed_dataset(train_set_path: str="", test_set_path: str="", save_to_folder: str="", validation_percantage: float=0.0):
    # load train dataframe
    train_dataframe = load_dataframe(train_set_path)

    # split train dataframe into train and validation dataframes
    val_size = train_dataframe.shape[0] * validation_percantage
    train_dataframe = train_dataframe.iloc[:int(train_dataframe.shape[0] - val_size), :]
    validation_dataframe = train_dataframe.iloc[int(train_dataframe.shape[0] - val_size):, :]

    # load test dataframe
    test_dataframe = load_dataframe(test_set_path)

    # preprocess datasets
    train_set = Preprocess(train_dataframe, batch_name="train-set").preprocess_dataset()
    test_set = Preprocess(test_dataframe, batch_name="test-set").preprocess_dataset()
    val_set = Preprocess(validation_dataframe, batch_name="val-set").preprocess_dataset()

    # save datasets
    write_json(train_set, file=(save_to_folder + "/train_set_tweets.json"))
    print("Saved train set.")
    write_json(test_set, file=(save_to_folder + "/test_set_tweets.json"))
    print("Saved test set.")
    write_json(val_set, file=(save_to_folder + "/val_set_tweets.json"))
    print("Saved validation set.")


""" remove labels from train- and validation-dataset """
def remove_labels(dataset: list) -> list:
    for i in range(len(dataset)):
        dataset[i] = dataset[i][0]

    return dataset


""" create vector embeddings for all words """
def word2vec(dataset_folder: str="", save_model_file: str="",):
    train_set = remove_labels(load_json(file=(dataset_folder + "/train_set_tweets.json")))
    test_set = load_json(file=(dataset_folder + "/test_set_tweets.json"))
    val_set = remove_labels(load_json(file=(dataset_folder + "/val_set_tweets.json")))
    full_data = np.concatenate((train_set, test_set, val_set))

    # create and save Word2Vec model
    model = Word2Vec(full_data, min_count=1, size=32)
    model.save(save_model_file)
    print("Saved word2vec model.")

    

if __name__ == "__main__":
    train_set_path, test_set_path, word2vec_model = "dataset/train.csv", "dataset/test.csv", "dataset/data/word_embeddings.model"

    create_preprocessed_dataset(train_set_path=train_set_path, test_set_path=test_set_path, save_to_folder="dataset/data", validation_percantage=0.1)
    word2vec(dataset_folder="dataset/data", save_model_file=word2vec_model)
