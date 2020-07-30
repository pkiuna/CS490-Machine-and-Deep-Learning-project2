
import matplotlib.pylab as plt
from keras.layers import Flatten, Conv2d, MaxPoolind2D, Lambda, MaxPool2D, Dense,Dropout, Activation,Lambda, MaxPool2D, batch_normalize,input
from keras.model import sequencial
from keras.utils import Categorical
from keras.utils import numpy_utils
from keras.models import sequencial, Load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import pandas
import random
import os
import cv2
from keras.preprocessing.image import imageToArray, load_Image
import numpy as np
from keras.constraint import maximum
from keras.optimizers import Adam
#loading dataset
df = pd.read_csv("monkeys-labels.txt")
print(df)
#labels
Columns = ['Label', 'Latin Name', 'Common Name', 'Train Images', 'Validation Images']
labels = pd.read_csv('monkeys-labels.txt')
labels = labels['Column Name']
print(labels)

height = 100
width = 100
channel = 2
batch_size = 40
seed = 1337
train_directory = Path('training')
test_directory = Path('validation')

  # Test Generator
testing_data = ImageDataGenerator(rescale=1./255)
testing_generator = testing_data.flow_from_directory(test_directory, target_size=(height,width),batch_size=batch_size, seed=seed)

  # Training Generator
training_data = ImageDataGenerator(rescale=1./255)
training_generator = training_data.flow_from_directory(train_directory_directory, target_size=(height,width),batch_size=batch_size, seed=seed)

def image_show(number_image, label):
    for k in range(number_image):
        image_dir = Path("training.txt" + label )
    print(image_dir)
    image_file = random.choice(os.listdir(image_dir) + label)
    print(image_file)
    image = cv2.imageRead("training.txt")
    print(image.shape)
    print(label)
    plt.figure(k)
    plt.title(image_file)
plt.show()

print(labels[4])
image_show(3, 'n2')
#reading images from folder
def read_data(path):
    images = []
    labels = []
    count = 1
for root, folder , file in os.walk(path):
    for c in file:
        filePath = os.path.join(root,c)
        image = load_Image(filePath,targetSize=(20,20))

#Loading images from Folders
read_data(train_directory)
read_data(test_directory)

#nomalize the data
x_train = x_train.astype('float32')
x_test = x_test.astype('Float32')

#scale
x_train = x_train / 255.0
x_test = x_test / 255.0

#encode data
y_train = np_utils.Categorical(y_train)
y_test = np_utils.Categorical(y_test)
numberOfClasses = y_test.shape[1]

#Model
model = sequencial()
model.add(Conv2D(32 (2, 2), inputShape=x_train.shape[2:], activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(32 (2, 2), activation='relu', padding='same'))
model.add(MaxPoolind2D(poolSize=(4,4)))
model.add(Conv2D(64 (2, 2), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(64 (2, 2), activation='relu', padding='same'))
model.add(MaxPoolind2D(poolSize=(4,4)))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(numberOfClasses,activation='softmax'))

#compiling model
epoch = 25
learningRate = 0.02
DC = learningRate/epoch

s_Gradient=SGD(lr=learningRate, DC=DC)

#fitting the model
history=model.fit(x_train, y_train, validateDate=(x_test,y_test), epoch=epoch, batch_size=32)

print(history.history.keys())
accuracy = history.history['Accuracy']
Validation_accuracy = history.history['Validation_ACC']
loss = history.history['Loss']
Validation_Loss = history.history['Validation Loss']
epoch = range(1, length(accuracy)+1)

plt.plot(epoch, accuracy, 'blue', label="Training Accuracy")
plt.plot(epoch, Validation_accuracy, 'red',label="Validation Accuracy")
plt.legend()
plt.figure()
plt.title("Training & validation loss")
plt.plot(epoch, Validation_Loss, 'blue', label="Validation Loss")
plt.plot(epoch, loss, 'blue', label="Training Loss")
plt.legend()
plt.show()

#saving model
model.save('Q3Model')

#loading model
model = loadModel('Q3Model')
print - (model.summary())

#Evaluating for prediction
scoring = modelEval(x_test,y_test,verbose=0)
print('Loss', scoring[0])
print('Accuracy ',scoring[1])

#making prediction
numbersToText = {
    0: "mantled_howler",
    1: "patas_monkey",
    2: "bald_uakari",
    3: "japanese_macaque",
    4: "pygmy_marmoset",
    5: "white_headed_capuchin",
    6: "silvery_marmoset",
    7: "commin_squirrel_monkey",
    8: "black_headed_night_monkey",
    9: "nilgiri_langur"
}

test_images = x_test
test_images = y_test

def Prediction(i):
    imageTest = test_images
    testData = x_test[[i]]

plt.title("Prediction ".format(numbersToText[model.predict_classes(testing_data)[0]]))
plt.show()



































