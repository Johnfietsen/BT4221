import numpy as np # linear algebra
import pandas as pd # test processing, CSV file I/O (e.g. pd.read_csv)
import re
from collections import Counter
import pickle
from datetime import datetime
import nltk
import preprocessor as prep
import string

def remove_twitter_specific(text):
    prep.set_options(
          prep.OPT.URL
        , prep.OPT.MENTION
        , prep.OPT.RESERVED
        , prep.OPT.EMOJI
        , prep.OPT.SMILEY)
    return prep.clean(text)

def remove_punctuation(text):
    text  = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    text = text.replace('“', '').replace('”', '').replace('…', '')
    return text.lower()

def stemming(text, stemmer):
    text_list =  [stemmer.stem(word) for word in text.split(' ')]
    #text = [word for word in text if word != ' ']
    return " ".join(text_list)

def clean_tweet(text, stemmer):
    text = remove_twitter_specific(text)
    text = remove_punctuation(text)
    text = stemming(text, stemmer)
    return text.strip()

# read train data
data = pd.read_csv('cleaned_data.csv', encoding='latin-1', header=None)

# give dataframe column names
data = data.rename(columns={0:"sentiment", 5:"tweet"})

# # create stemmer
# stemmer = nltk.PorterStemmer()
#
# print("start cleaning")
# # clean tweets remove uppercasing and punctuations
# # data['tweet'] = data['tweet'].apply(lambda x: x.lower())
# # data['tweet'] = data['tweet'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))
# data['tweet'] = data['tweet'].apply(lambda x: clean_tweet(x, stemmer))


print(data['sentiment'].value_counts())


# positive=data[data['sentiment']==4]
# negative=data[data['sentiment']==0].sample(n=positive.shape[0])
# final=pd.concat([positive,negative])

# final.to_csv("equal.csv", header=False, index=False)
