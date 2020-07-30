import pandas as pd
from keras_processing.text import Tokenizer
from sklearn import preprocessing
from keras.models import Sequential
from keras_processing.sequence import pad_sequences
from keras.layers import Dense, Dropout, Conv1d, GlobalMaxPooling1D
from sklearn.model_selection import train_train_split
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers import embedding
from tensorflow.python.keras.utils.no_utils import Caterogical

 #Load Data-set
Train = pd.read_csv("train.tsv.zip")

#assigning values
A = Train["Phrase"].values
B = Train["Sentiment"].values

#Tokenization
tokenizer = Tokenizer(number_Words=2000)
tokenizer.fitTest(x)
A = tokenizer. textSentences
A= pad_sequences(A)

#encode
Length = preprocessing.LabelEncoder()
b = Length.fit_transform(b)
b = Caterogical(b)

# Testing
X_train, x_test, y_train, y_test = train_train_split(A,b,test_size = 0.5, random=2000)

#Convolutional Neural Network Layers
model = Sequential()
model.add(embedding(2000,A.shape[1]))
model.add(Dropout(0.5))
model.add(Conv1d(filter=20, kernelSize=5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(b.shape[1], activation = 'softmax'))
model.compile(loss='categorical_crossentropy', optemizer='adam', metrics=["Accuracy"])
history = model.fit (x_train, y_train, eposh=10, verbose=true, validationData = (x_test, y_test))

#gathering accuracy score

ACC_score = model.evaluate(x_test,y_test)
print("Accuracy Model " + str((ACC_score[1]*100)))

#plotting loss
plt.plot(history.history['Loss'])
plt.plot(history.history['Validation Loss'])
plt.title('CNN model loss')
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.show()









