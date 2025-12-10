'''
frame.py

Define the types of frames and the function to process them.

'''

import cv2
import numpy as np

class Frame:
    def __init__(self, frame_ndarray: np.ndarray):
        self.frame = frame_ndarray