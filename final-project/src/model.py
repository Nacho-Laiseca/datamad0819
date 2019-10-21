from src import CONST
import numpy as np
import cv2
import os
import time
from imutils import paths
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from keras import applications
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
import matplotlib.pyplot as plt
from PIL import Image
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import pandas as pd


# PREPROCESSING DATASET -----------------
print("Loading dataset...")

#grab the paths to our input images followed by shuffling them 
imagePaths = sorted(list(paths.list_images('dataset')))


lookup = dict()
reverselookup = dict()
count = 0
for j in os.listdir('dataset'):
    if not j.startswith('.'): # If running this code locally, this is to 
        print(j)                      # ensure you aren't reading in hidden folders
        lookup[j] = count
        reverselookup[count] = j
        count = count + 1


data = []
labels = []
lookup = {}
for imagePath in imagePaths:
    # load the image, pre-process it, resize it to IMAGE_SIZE, store it in the data list
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (CONST.IMAGE_SIZE, CONST.IMAGE_SIZE))
    image = img_to_array(image)
    data.append(image)
    label = imagePath.split(os.path.sep)[-2]
    folders = []
    for folder in os.listdir('dataset'):
        if not folder.startswith('.'):
            folders.append(folder)
    for i in range(len(folders)):
        if label == folders[i]:
            label = i
            labels.append(label)
            if folders[i] not in lookup.keys():
                lookup[i]= folders[i]
            else:
                pass
    


# scaling the data points from [0, 255] to the range [0, 1]
data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)



from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(data,labels,test_size = 0.1)
X_train[0].shape


# convert the labels from integers to vectors
NUM_CLASSES = len(folders)

y_train = to_categorical(y_train).astype(int)
y_test = to_categorical(y_test).astype(int)


def save_bottlebeck_features():
    
    print("Training the VGG16 bottleneck features ...")
    
    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights=None, input_shape=(100,100,1)) 
    
    bottleneck_features_train = model.predict(X_train, verbose=1)
    np.save('bottleneck_features_train.npy', bottleneck_features_train)
    
    bottleneck_features_validation  = model.predict(X_test, verbose=1)
    np.save('bottleneck_features_validation.npy', bottleneck_features_validation)


save_bottlebeck_features()


print("[INFO] training the top of the model...")


#save_bottlebeck_features()

train_data = np.load('bottleneck_features_train.npy')
validation_data = np.load('bottleneck_features_validation.npy')


from keras import layers

top_model = Sequential()
top_model.add(Flatten(input_shape=train_data.shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.6))
top_model.add(Dense(3, activation='sigmoid'))


top_model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


EPOCHS = 50
BATCH_SIZE = 2
NUM_CLASSES = 3
TEST_SIZE = 0.25


start = time.time()

model_info = top_model.fit(train_data, y_train,batch_size=BATCH_SIZE,epochs=EPOCHS,
                           validation_data=(validation_data, y_test),
                           verbose=1)
end = time.time()

print ("\nModel training time: %0.1fs\n" % (end - start))


scores = top_model.evaluate(validation_data, y_test)
print("\nTest Loss:  %.2f%%" % (scores[0]*100))
print("Test Accuracy: %.2f%%\n" % (scores[1]*100))


# Saving model
'''if not os.path.isdir(CONST.SAVE_DIR):
    os.makedirs(CONST.SAVE_DIR)
model_path = os.path.join(CONST.SAVE_DIR, CONST.BOTTLENECK_MODEL)
top_model.save(model_path)
print('\nSaved trained model at %s ' % model_path)'''



