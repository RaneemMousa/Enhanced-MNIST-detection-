# first we import the data
# what makes emnist different ? what file did we use and why ?
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.show()
# we will choose the digits dataset since we want a classification of digits
train  = pd.read_csv(r'D:\raneem\ML\deep learning\emnist-digits-train.csv', delimiter = ',')
test =
mapp  = pd.read_csv(r'D:\raneem\ML\deep learning\empd.read_csv(r'D:\raneem\ML\deep learning\emnist-digits-test.csv', delimiter = ',')nist-digits-mapping.txt', delimiter = ',')
#print(train.head())
train.info()
# since we dont have much of a featurs here we should try at least to come up with the images that we are classifying
# but first let us define our vaiables and input / split the data
trainData = train.values
testData = test.values
# . values gives back the numpy array so we dont mix it with the panda series
x_train = trainData[:, 1:].astype('float32')
y_train = trainData[:, 0:1]

x_test = testData[:, 1:].astype('float32')
y_test = testData[:, 0:1]
#normalizing the data
x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
# lets try to plot the image
j=10
m=x_train.shape[0]
X=x_train.reshape((m,28,28,1))
#plt.figure(1)
#plt.gray()
#plt.imshow(X[j].reshape((28,28)))
#plt.show()
#llllooooollllliiiishhhhh
# preprocessing
# first we have no null values
# we need to normalize the data using one hot encoding or scale the data
# scalling data
#from sklearn.preprocessing import MinMaxScaler
#scaler = MinMaxScaler()
#scaler.fit(x_train)
#x_train = scaler.transform(x_train)
#x_test = scaler.transform(x_test)
# reshape fo NN
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
# now we create a model and train it
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D,AveragePooling2D
import sklearn.metrics as metrics
train_x, val_x, train_y, val_y = train_test_split(x_train, y_train, test_size= 0.10, random_state=7)
#model = Sequential()
#model.add(Dense(units=784,activation='relu'))
#model.add(Dense(units=500,activation='relu'))
#model.add(Dense(units=250,activation='relu'))
#model.add(Dense(units=1,activation='sigmoid'))
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# we will use a dense of 784

print(x_train.shape)
# to avoid overfitting we will use two methods to avoid that
# 1 early stopping
#model = Sequential()
#model.add(Dense(units=784,activation='relu'))
#model.add(Dense(units=784,activation='relu'))
#model.add(Dense(units=500,activation='relu'))
#model.add(Dense(units=250,activation='relu'))
#model.add(Dense(units=1,activation='sigmoid'))
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#from tensorflow.keras.callbacks import EarlyStopping
#early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
#model.fit(x=x_train,
         # y=y_train,
         # epochs=50,
         # validation_data=(x_test, y_test), verbose=1,
          #callbacks=[early_stop]
         # )

#model_loss = pd.DataFrame(model.history.history)
#model_loss.plot()
#plt.show()
#something is wrong , lets try dropout its probably the data normalization
model = Sequential()

model.add(Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
model.add(AveragePooling2D())

model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
model.add(AveragePooling2D())

model.add(Flatten())

model.add(Dense(units=120, activation='relu'))

model.add(Dense(units=84, activation='relu'))

model.add(Dense(units=10, activation = 'softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train , epochs=20, batch_size=512, verbose=1, \
                    validation_data=(val_x, val_y))
def plotgraph(epochs, acc, val_acc):
    # Plot training & validation accuracy values
    plt.plot(epochs, acc, 'b')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(acc)+1)
plotgraph(epochs, acc, val_acc)
