

from sklearn.datasets import fetch_20newsgroups
from keras.preprocessing.sequence import pad_sequences
import keras
from keras.layers import Embedding, Dense, LSTM, GRU
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import string

categories = ['comp.sys.mac.hardware',
 'comp.windows.x',
 'misc.forsale',
 'rec.autos',
'soc.religion.christian',
 'talk.politics.guns',
 'talk.politics.mideast']

#train and test Data
data = fetch_20newsgroups(shuffle=True, subset='train', categories=categories)
test = fetch_20newsgroups(shuffle=True, subset='test', categories=categories)

#tokenize sentences
words = 200000
tokenizer = Tokenizer(numberOfWords=words)
tokenizer.fitToText(news)
sequences = Tokenizer.textsToSequences(tokenizer, news)

WordIndex = tokenizer.WordIndex
print('tokens.' % len(WordIndex))
maxLength = 2000
X = pad_sequences(sequences, maxlen=maxLength)
Y = keras.utils.to_categorical(news_topics)

print('Data tensor:', X.shape)
print('Label tensor:', Y.shape)

X_train, xValidity, y_train, yValidity = train_test_split(X, Y, test_size=0.2)
len(word_index)

#embedding
embedding_length = 64
model = Sequential()
model.add(LSTM(100, return_sequences=True,input_shape=(2, maxLength)))
model.add(LSTM(100, dropout_W=0.25))
model.add(Dense(5, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['ACC'])
print(model.summary())

import numpy as np
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
xValidity = np.reshape(xValidity, (xValidity.shape[0], 1, xValidity.shape[1]))

history=model.fit(X_train, y_train, epochs=20, verbose=True, validation_data=(xValidity, yValidity), batch_size=64)

Loss,Accuracy = model.evaluate(xValidity, yValidity)
print("Loss before embedding layer :", Loss)
print("Accuracy before adding embedding layer:",Accuracy)
#adding an embedding layer
model1 = Sequential()
model1.add(Embedding(len(word_index), embedding_length, input_length=max_length))
model1.add(LSTM(100, dropout_W=0.25))
model1.add(Dense(7, activation='softmax'))
model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model1.summary())

history1=model1.fit(X_train1, y_train1, epochs=20,verbose=True, validation_data=(X_validOne,y_validOne), batch_size=64)

Loss,Accuracy = model1.evaluate(X_validOne, y_validOne)
print("Loss after embedding :", Loss)
print("Accuracy after embedding layer:",Accuracy)

plt.plot(history1.history['ACC'])
plt.plot(history1.history['Validity Accuracy'])
plt.title('Train & validation acc')
plt.xlabel('epochs')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='bottom right')
plt.show()

plt.plot(history1.history['loss'])
plt.plot(history1.history['validation loss'])
plt.title('Train & validation loss')
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()















