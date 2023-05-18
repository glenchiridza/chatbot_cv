import random
import json
import pickle
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

words = sorted(set(words))

# saving the words and classes list to binary files ( Serialization )
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
output_empty = [0] * len(classes)
for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(
        word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])
random.shuffle(training)
training = np.array(training)

train_x = list(training[:, 0])
train_y = list(training[:, 1])


