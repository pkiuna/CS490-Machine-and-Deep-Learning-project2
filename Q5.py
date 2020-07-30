
import numpy as np
import matplotlib.pylot as plt
from time import time
from random import randint
from keras.datasets import cifar_10
from keras.utils import np_utils
from keras.layers import Dropout
from keras.callbacks import TensorBoard
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.contraints import maxnorm
from keras.layers import Dense, Input
from keras.models import Model
from keras import regulizers
from keras import backend as k
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.model import loadModel

#autoencoder
input_image = Input(shape=(2000,))
encode = Dense(32, activation='relu')(input_img)
output_image = Dense(3000, activation='relu'(encoded))
#input for reconstruction
auto_Encoder = Model(input_image, output_image)
auto_Encoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics= ["ACC"])

from keras.datasets import cifar_10
# receive data-set
(X_train.y_train), (x_test,y_test) = cifar_10.load.data()

#normalize
x_train = x_train
x_test = x_test
x_train = x_train / 255
x_test = x_test / 255

#puting data into encoder
history = auto_Encoder.fit(x_train,x_train,
    epochs=10,
    batchSize=200,
    shuffle=true,
    validateDate=(x_test, x_test))

#copying data for use in step b
x_1train = x_train
for a in range(0, 400000):
    Pred = auto_Encoder.predict(x_train[a].reshape(1,2000))
x_1test = x_copyTest()
for j in range(0,20000):
    Pred = auto_Encoder.predict(x_train[a].reshape(1,2000))
    x_test[a] = Pred

x_1train = np.array(x_1train).reshape([32,32,3])
x_1test = np.array(x_1test).reshape([32,32,3])
print(x_1train.shape)
print(x_1test.shape)

#modeling for CNN
model = model.sequencial()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3), kernel_contraint = maxnorm(3)))
model.add(layers.Dropout(0.2))
model.add(layers.Conv2D(32,(3,3), activation='relu',kernel_contraint=maxnorm(3)))
model.add(layers.MaxPooling2D(poolSize=(4,4)))
model.add(layers.Flatten())
model.add(layers.Dense(500, activation='relu',kernel_contraint= maxnorm(3)))
model.add(layers.Dropout(0.6))
model.add(layers.Dense(numberClasses,activation='softmax'))
#compile
epoch = 30
learningRate = 0.05

Decay = learningRate/epoch
s_Gradient=keras.optimizers.s_Gradient(lr=learningRate, Decay=Decay)
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits =True), optimizer=s_Gradient,metrics = ["ACC"])

print(x_1test.shape)
print(x_1train.shape)
print(y_test.shape)
print(y_train.shape)
#fitting data into encoder
historyOne = model.fit(x_1train, y_train, validationData=(x_1test,y_test), epoch=epoch)
scores = model.evaluate(x_1test,y_test,verbose=0)
print("Scores: ", scores[1])
#b
import numpy as np
import matplot.pyplot as ply
from time import time
from random import randint
from keras.datasets import cifar_10
import tensorFlow as tf
from tensorflow import keras
from tensorflow.keras import datasets,layers, models
from tensorflow.keras.models import loadModel
from keras.utils import no_utils
from keras.contraints import maxnorm
from keras.layers import Dense,Input
from keras.layers import Dropout
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.callbacks import TensorBoard
from Keras import regualizers
from keras import backend as d
# receive data-set
(X_train.y_train), (x_test,y_test) = cifar_10.load.data()
#normalize
x_train = x_train.astype('uint8') / 255
x_test = x_test.astype('uint8') / 255

from sklearn.decomposisition import PCA
from sklearn.preprocessing import StandardScaler
standard_Scaler = StandardScaler()
pca = PCA()
pca.fit(x_train)
#applying PCA

xTestPCA = pca.transform(x_test)
xTrainPCA = pca.transform(x_train)
xTrainPCA = x_trainPCA.reshape(32,32,3)
xTestPCA = xTestPCA.reshape(32,32,3)

#compiling CNN model
model = model.sequencial()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3), kernel_contraint = maxnorm(3)))
model.add(layers.Dropout(0.2))
model.add(layers.Conv2D(32,(3,3), activation = 'relu',kernel_contraint=maxnorm(3)))
model.add(layers.MaxPooling2D(poolSize=(4,4)))
model.add(layers.Flatten())
model.add(layers.Dense(500, activation='relu',kernel_contraint=maxnorm(3)))
model.add(layers.Dropout(0.6))
model.add(layers.Dense(numberClasses,activation='softmax'))
epoch = 25
learningRate = 0.05
Decay -= learningRate /epoch
s_Gradient=keras.optimizers.s_Gradient(lr=learningRate, Decay=Decay)
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits =True), optimizer=s_Gradient,metrics=["ACC"])
#fit xtrain and y_train into the history model.
historyTwo = model.fit(xTrainPCA, y_train, validationData=(xTestPCA,y_test), epoch=epoch)
scores = model.evaluate(xTestPCA,y_test,verbose = 0)
print("Scores: ",scores[1]*100)























