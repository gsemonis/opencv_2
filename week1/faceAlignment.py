# Copyright 2017 BIG VISION LLC ALL RIGHTS RESERVED
#
# This code is made available to the students of
# the online course titled "Computer Vision for Faces"
# by Satya Mallick for personal non-commercial use.
#
# Sharing this code is strictly prohibited without written
# permission from Big Vision LLC.
#
# For licensing and other inquiries, please email
# spmallick@bigvisionllc.com
#
import sys
import cv2
import dlib
import numpy as np
import faceBlendCommon as fbc

if __name__ == '__main__':

  # Landmark model location
  PREDICTOR_PATH = "./data/models/shape_predictor_5_face_landmarks.dat"

  # Get the face detector
  faceDetector = dlib.get_frontal_face_detector()
  # The landmark detector is implemented in the shape_predictor class
  landmarkDetector = dlib.shape_predictor(PREDICTOR_PATH)

  # Read image
  im = cv2.imread("./data/images/face1.png")
  imDlib = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

  # Detect landmarks.
  points = fbc.getLandmarks(faceDetector, landmarkDetector, imDlib)

  points = np.array(points)

  # Convert image to floating point in the range 0 to 1
  im = np.float32(im)/255.0

  # Dimensions of output image
  h = 600
  w = 600

  # Normalize image to output coordinates.
  imNorm, points = fbc.normalizeImagesAndLandmarks((h, w), im, points)

  imNorm = np.uint8(imNorm*255)

  cv2.imshow("Original Face", im)
  cv2.imshow("Aligned Face", imNorm)
  cv2.imwrite("AlignedFace.jpg", imNorm)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
