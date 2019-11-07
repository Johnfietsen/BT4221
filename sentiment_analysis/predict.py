import numpy as np # linear algebra
import pandas as pd # test processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
import re
from collections import Counter
import pickle
import glob
import csv


def main():

    # give folder name for pretraine files
    folder_pretrained = "run1"

    # load all the data
    path = "../data/tweets/stemmed/*.csv"
    files = glob.glob(path)

    # load tokenizer
    print("Loading tokenizer")
    with open('results/{}/tokenizer.pickle'.format(folder_pretrained), 'rb') as handle:
        tokenizer = pickle.load(handle)

    # load json and create model
    json_file = open('results/{}/model.json'.format(folder_pretrained), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights("results/{}/model.h5".format(folder_pretrained))
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # save negative and postitive counter in a dict
    scores = {}

    # for every file predict positive and negative tweets
    for file in files:
        print("predicting score for file: " + file)

        # read tweets
        data = pd.read_csv(file, encoding='latin-1', header=None)

        # give dataframe column names
        data = data.rename(columns={0:"tweet"})

        # make sure all tweets are strings
        data['tweet'] = data['tweet'].apply(lambda x: str(x))

        # create one hot vectors from sequence
        X = tokenizer.texts_to_sequences(data['tweet'].values)

        # pad sequences with same length as training
        X = pad_sequences(X, maxlen=40)

        # predict setiment of tweets
        predictions = loaded_model.predict(X)

        # count negative and postive tweets
        negative, positive = count_pos_neg(predictions)

        # save counts in dict
        scores[file] = (negative, positive)

    # write to a csv
    create_csv(scores)

def create_csv(scores):
    with open('results/score.csv', 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',')
        filewriter.writerow(["File", "Negative", "Positive", "ratio neg/pos"])
        for score in scores:
            file = score.split("/")[-1]
            filewriter.writerow([file, scores[score][0], scores[score][1],  scores[score][0]/scores[score][1]])

def count_pos_neg(predictions):
    negative = 0
    positive = 0
    for prediction in predictions:
        if prediction[0] > prediction[1]:
            negative += 1
        else:
            positive += 1
    return negative, positive

if __name__ == "__main__":
    main()
