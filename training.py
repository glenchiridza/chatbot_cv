import _random
import json
import _pickle
import numpy as np
import nltk

from keras.models import Sequential
from nltk.stem import WordNetLemmatizer
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()

intents = json.loads(open("intents.json").read())

words = []
classes = []
documents = []
ignore_letters = ["?", "!", ".", ","]

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # seperate words from patterns
        word_list = nltk.word_tokenize(pattern)
        # add them to words list
        words.extend(word_list)
        # associate patterns with respective tags
        documents.append((word_list, intent['tag']))
        # append the tags to the class list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# store the root words
words = [lemmatizer.lemmatize(word)
         for word in words if word not in ignore_letters]
