# -*- coding: utf-8 -*-
import os
import cv2
# Global variables

RANDOM_SEED = 2017

FRAMES_PER_VIDEO = 100
IMAGE_SIZE = 100

SAVE_DIR = os.path.join(os.getcwd(), 'saved_models')
BOTTLENECK_MODEL = 'bottleneck_model.h5'

FONT = cv2.FONT_HERSHEY_SIMPLEX