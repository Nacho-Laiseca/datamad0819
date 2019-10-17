from src import CONST
import numpy as np
import cv2
from keras.models import load_model
import os
from keras.preprocessing.image import img_to_array
from keras import applications
from pygame import mixer
import time


model = applications.VGG16(include_top=False, weights=None, input_shape=(CONST.IMAGE_SIZE,CONST.IMAGE_SIZE,3)) 
top_model = load_model(os.path.join(CONST.SAVE_DIR,CONST.BOTTLENECK_MODEL))

labels = [x for x in os.listdir('./dataset') if not x.startswith(".")]

def predictImage(imagepath):
    image = cv2.imread(imagepath)
    frame2 = cv2.resize(image, (CONST.IMAGE_SIZE, CONST.IMAGE_SIZE))
    frame2 = img_to_array(frame2)
    frame2 = np.array(frame2, dtype="float32") / 255
    y_pred = top_model.predict_classes(model.predict(frame2[None,:,:,:]))
    return y_pred

predictImage(input("\nPath to image: "))


print("\nImporting preferences and calibrating virtual environment...")
time.sleep(2)
mixer.init()
mixer.music.load('sounds/jarvis_uploaded.wav')
mixer.music.play()

predictImage(input("\nPath to image: "))

os.system('spotify open')

predictImage(input("\nPath to image: "))

os.system('spotify play uri spotify:track:08mG3Y1vljYA6bvDt4Wqkj')