import random
import json
import pickle
import numpy as np
import nltk
from keras.models import load_model
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

# initializing classes and file contents
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_glen.model')


# seperates words from sentences given as input
def separate_sentences(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word)
                      for word in sentence_words]
    return sentence_words


# bag of words, append 1 to list if the word contained inside input given

def bag_of_words(sentence):
    sentence_words = separate_sentences(sentence)
    bag = [0] * len(words)
    for input_word in sentence_words:
        for i, word in enumerate(words):
            # check existence of word in input
            if word == input_word:
                # if present assign 1 to the bag within position i--index of the word found
                bag[i] = 1
    return np.array(bag)
