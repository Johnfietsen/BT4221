import numpy as np # linear algebra
import pandas as pd # test processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
import re
from collections import Counter
import pickle

def create_plot(data, xlabel, ylabel):
    plt.plot(data)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.show()


# read train data
train = pd.read_csv('train.csv', encoding='latin-1', header=None)
train = train[[train.columns[0], train.columns[5] ]]

# read test data
test = pd.read_csv('test.csv', encoding='latin-1', header=None)
test = test[[test.columns[0], test.columns[5] ]]

# give dataframe column names
train = train.rename(columns={0:"sentiment", 5:"tweet"})
test = test.rename(columns={0:"sentiment", 5:"tweet"})

# clean tweets remove uppercasing and punctuations
train['tweet'] = train['tweet'].apply(lambda x: x.lower())
train['tweet'] = train['tweet'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))
test['tweet'] = test['tweet'].apply(lambda x: x.lower())
test['tweet'] = test['tweet'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))

# count unique words
word_count = Counter(" ".join(train['tweet'].values.tolist()).split(" "))

# find number of unique word, minus 1 for ''
nr_words = len(word_count) - 1

# fit one-hot encoding on train data
# TODO:Choose nr_words!
tokenizer = Tokenizer(num_words=nr_words, split=' ')
tokenizer.fit_on_texts(train['tweet'].values)

# create (x,y) for train data
X = tokenizer.texts_to_sequences(train['tweet'].values)
X = pad_sequences(X)
Y = pd.get_dummies(train['sentiment']).values

# take part of train vor validation
X_train, X_val, Y_train, Y_val = train_test_split(X,Y, test_size = 0.2, random_state = 42)


# create (x,y) for test data
X_test = tokenizer.texts_to_sequences(test['tweet'].values)
X_test = pad_sequences(X_test)
Y_test = pd.get_dummies(test['sentiment']).values

print("Train:", X_train.shape,Y_train.shape)
print("Test: ", X_test.shape,Y_test.shape)


# Hyperparameters, LOOK INTO THIS
embed_dim = 128
lstm_out = 196
dropout = 0.2
batch_size = 32
epochs = 5
nr_words = nr_words

# define model
model = Sequential()
model.add(Embedding(nr_words, embed_dim,input_length = X_test.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=dropout, recurrent_dropout=0.2))
model.add(Dense(3,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose = 2, validation_data=(X_val, Y_val))
print("history:", history.history)

create_plot(history.history["val_loss"], "Epochs", "Validation loss")
create_plot(history.history["val_acc"], "Epochs", "Accuracy")


results = model.evaluate(X_test, Y_test, verbose=2, batch_size=batch_size)
print("results: ", results)

# save tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# save model
print("Saving model")
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
