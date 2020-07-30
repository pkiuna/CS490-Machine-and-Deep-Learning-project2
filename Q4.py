
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras.utils as ku

from tensorflow import set_random_seed
from numpy.random import seed
set_random_seed(2)
seed(1)

import pandas as pd
import numpy as np
import string, os
import warnings

import tensorflow as tf
tf.random.set_seed(2)
import pandas as pd
import numpy as np
import string, os

directory = "newyork_headline.zip"
headlines = []
for filename in os.listdir(directory):
    if 'Articles' in filename:
        print(filename)
        articleData = p.read_csv(directory + filename)
        headlines.extend(list(articleData.headlines.values))

len(headlines)
all_headlines = [h for h in all_headlines if h != "Unknown"]
len(all_headlines)

#Process Data
def clean_data(text):
    text = "".join(v for v in text if v not in string.punctuation).lower()
    text = text.encode("utf8").decode("ascii",'ignore')
    return text

corpus = [clean_data(headline) for headline in all_headlines]
corpus[:10]

#tokenization
tokenizer = Tokenizer()

def get_sequence_of_tokens(corpus):
    q = 0
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1

    if q == 0:
        print(total_words)

    # convert data into sequence of tokens
    input_sequences = []
    for line in corpus:
        if q == 0:
            print(line)

        token_list = tokenizer.texts_to_sequences([line])[0]

        if q == 0:
            print(token_list)
            print(len(token_list))
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i + 1]
            if q == 0:
                print(n_gram_sequence)
            input_sequences.append(n_gram_sequence)

            if q == 0:
                print(input_sequences)

            q = 1
        return input_sequences, total_words
    input_sequence, total_words = get_sequence_of_tokens(corpus)
    print(total_words)


    #pad sentences and make them equal
def generate_padded_sequences(input_sequence):
    max_sequence_len = max([len(x) for x in input_sequence])
    input_sequences = np.array(pad_sequences(input_sequence, maxlen=max_sequence_len, padding='pre'))
    predictors, labels = input_sequences[:, :-1], input_sequences[:, -1]
    labels = ku.to_categorical(labels, num_classes=total_words)

    return predictors, labels, max_sequence_len

    predictors, labels, max_seq_len = generate_padded_sequences(input_sequence)

 #Creating Model
def create_model(maxSequenceLength, total_words):
    input_len = maxSequenceLength - 1

    model = Sequential()

    # input: embedding layer
    model.add(Embedding(total_words, 10, input_length=input_len))

    # hidden: lstm layer
    model.add(LSTM(100))
    model.add(Dropout(0.1))
    # output layer
    model.add(Dense(total_words, activation='softmax'))

    # compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model
model = create_model(max_seq_len, total_words)
model.summary()

# fitting (training) the model by passing predictors and labels as training data
model.fit(predictors, labels, epochs=100, verbose=5)

#generate the text to predict headline
def generate_text(seed_text, next_words, model, max_seq_len):
    w = 0
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        if w == 0:
            print(token_list)
        token_list = pad_sequences([token_list], maxlen=max_seq_len - 1, padding='pre')

        if w == 0:
            print(token_list)

        predicted = model.predict_classes(token_list, verbose=0)
        if w == 0:
            print(predicted)
        output_word = ''

        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text = seed_text + " " + output_word
        w = 1
    return seed_text.title()
print(generate_text("united states", 5, model, max_seq_len))







