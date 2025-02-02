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
from datetime import datetime
import os
import argparse

# default settings
FLAGS = None
EMBED_DIM_DEFAULT = 128
LSTM_OUT_DEFAULT = 128
FOLDER_PRETRAINED_DEFAULT = "run1"
DATA_FILE_DEFAULT = "sanders.csv"
PRETRAIN_DEFAULT = False
BATCH_SIZE_DEFAULT = 32
EPOCHS_DEFAULT = 20

def main():
    # Hyperparameters, LOOK INTO THIS
    embed_dim = FLAGS.embed_dim
    lstm_out = FLAGS.lstm_out
    dropout = 0.2
    test_split = 0.2
    val_split = 0.2

    # for sentiment140 set batch_size==256, for smaller datasets set smaller!
    batch_size = FLAGS.batch_size
    epochs = FLAGS.epochs
    # size of "vocabulary" for one hot vectors
    nr_words = 2000

    # give file for training data and possible pretrained model
    data_file = FLAGS.data_file
    folder_pretrained = FLAGS.folder_pretrained
    pretrain = FLAGS.pretrain

    # read train datals
    data = pd.read_csv("data/{}".format(data_file), encoding='latin-1', header=None)

    # give dataframe column names
    if data_file == "sanders.csv" or data_file == "semeval_balanced.csv" or data_file == "semeval_sanders.csv":
        data = data.rename(columns={0:"sentiment", 1:"tweet"})
        batch_size = 32
    else:
        data = data.rename(columns={0:"sentiment", 5:"tweet"})
        batch_size = 256



    # make sure every tweet is a string
    data['tweet'] = data['tweet'].apply(lambda x: str(x))



    if pretrain == False:
        # fit one-hot encoding on train data
        tokenizer = Tokenizer(num_words=nr_words, split=' ')
        tokenizer.fit_on_texts(data['tweet'].values)
    else:
        with open('results/{}/tokenizer.pickle'.format(folder_pretrained), 'rb') as handle:
            tokenizer = pickle.load(handle)

    # create (x,y) for train data
    X = tokenizer.texts_to_sequences(data['tweet'].values)
    X = pad_sequences(X, maxlen=40)
    Y = pd.get_dummies(data['sentiment']).values

    # take part of train vor validation
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = test_split, random_state = 42)

    print("Train: ", X_train.shape,Y_train.shape)
    print("Test: ", X_test.shape,Y_test.shape)

    # create model either new or pretrained
    if pretrain == False:
        print("Creating new model")
        #define model
        model = Sequential()
        model.add(Embedding(nr_words, embed_dim,input_length = X_test.shape[1]))
        model.add(SpatialDropout1D(0.4))
        model.add(LSTM(lstm_out, dropout=dropout, recurrent_dropout=0.2))
        model.add(Dense(2,activation='softmax'))
    else:
        print("Using pretrained model")
        # load json and create model
        json_file = open('results/{}/model.json'.format(folder_pretrained), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)

        # load weights into new model
        model.load_weights("results/{}/model.h5".format(folder_pretrained))
        print("Loaded model from disk")

    # evaluate loaded model on test data
    model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # train model and evaluate
    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose = 1, validation_split=val_split)
    results = model.evaluate(X_test, Y_test, verbose=2, batch_size=batch_size)

    # print results
    print("history:", history.history)
    print("results: ", results)

    # get timestamp
    timestamp = datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")

    # Create folder to save model and plots
    if not os.path.exists("results"):
        os.mkdir("/results")


    if not os.path.exists(timestamp):
        os.mkdir("results/{}".format(timestamp))
        print("Directory " , timestamp,  " Created ")
    else:
        print("Directory " , timestamp ,  " already exists")


    # create plots
    create_plot(history.history["val_loss"], history.history["loss"], "Epochs", "Validation loss", "loss_", timestamp)
    create_plot(history.history["val_acc"], history.history["acc"], "Epochs", "Accuracy", "accuracy_", timestamp)

    # save tokenizer
    with open("results/{}/tokenizer.pickle".format(timestamp), 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # save model
    print("Saving model")
    model_json = model.to_json()
    with open("results/{}/model.json".format(timestamp), "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights("results/{}/model.h5".format(timestamp))
    print("Saved model to disk")
    write_settings(embed_dim, lstm_out, history, results, timestamp, pretrain, data_file, folder_pretrained, batch_size, epochs)

def create_plot(data_val, data_train, xlabel, ylabel, title, timestamp):
    plt.plot(data_val, label="val")
    plt.plot(data_train, label="train")
    plt.legend()
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.savefig("results/{}/{}.png".format(timestamp, title))
    plt.close()

def write_settings(embed_dim, lstm_out, history, results, timestamp, pretrain, data_file, folder_pretrained, batch_size, epochs):
    file = open("results/{}/settings_results.txt".format(timestamp),"w")
    file.write("embed: " + str(embed_dim) + "\n")
    file.write("lstm_out: " + str(lstm_out) + "\n")
    file.write("pretrained: " + str(pretrain) + "\n")
    file.write("data_file: " + str(data_file) + "\n")
    file.write("batch_size: " + str(batch_size) + "\n")
    file.write("folder_pretrained: " + str(folder_pretrained) + "\n")
    file.write("epochs: " + str(epochs) + "\n")
    file.write(str(history.history) + "\n")
    file.write(str(results))
    file.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--embed_dim', type = int, default = EMBED_DIM_DEFAULT,
                        help='embedding size lstm')
    parser.add_argument('--lstm_out', type = int, default = LSTM_OUT_DEFAULT,
                        help='size fully connected layer lstm')
    parser.add_argument('--data_file', type = str, default = DATA_FILE_DEFAULT,
                        help='data file')
    parser.add_argument('--folder_pretrained', type = str, default = FOLDER_PRETRAINED_DEFAULT,
                        help='pretrained model')
    parser.add_argument('--pretrain', type= bool, default= PRETRAIN_DEFAULT,
                        help='continue training using old model')
    parser.add_argument('--batch_size', type= int, default= BATCH_SIZE_DEFAULT,
                        help='batch size')
    parser.add_argument('--epochs', type= int, default= EPOCHS_DEFAULT,
                        help='epochs')
    FLAGS, unparsed = parser.parse_known_args()

    main()
