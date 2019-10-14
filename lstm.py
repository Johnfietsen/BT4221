import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
import re
from collections import Counter

# read data
data = pd.read_csv('test.csv', encoding='latin-1', header=None)
data = data[[data.columns[0], data.columns[5] ]]

# give dataframe column names
data = data.rename(columns={0:"sentiment", 5:"tweet"})

# clean tweets remove uppercasing and punctuations
data['tweet'] = data['tweet'].apply(lambda x: x.lower())
data['tweet'] = data['tweet'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))

# count unique words
word_count = Counter(" ".join(data['tweet'].values.tolist()).split(" "))

# find number of unique word, minus 1 for ''
nr_words = len(word_count) - 1

# create tokinizer and create one-hot vectors
tokenizer = Tokenizer(num_words=nr_words, split=' ')
tokenizer.fit_on_texts(data['tweet'].values)
X = tokenizer.texts_to_sequences(data['tweet'].values)
X = pad_sequences(X)

# Hyperparameters, LOOK INTO THIS
embed_dim = 128
lstm_out = 196
dropout = 0.2

# define model
model = Sequential()
model.add(Embedding(nr_words, embed_dim,input_length = X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=dropout, recurrent_dropout=0.2))
model.add(Dense(3,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])

# create target variables
Y = pd.get_dummies(data['sentiment']).values

# create test and train sets
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.33, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)


batch_size = 32
model.fit(X_train, Y_train, epochs = 7, batch_size=batch_size, verbose = 2)
