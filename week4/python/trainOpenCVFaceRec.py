# Let's import necessary modules
import os, sys, dlib
import cv2
import math
import numpy as np
import faceBlendCommon as fbc
# import cPickle
try:
  import cPickle  # Python 2
except ImportError:
  import _pickle as cPickle  # Python 3

faceWidth = 64
faceHeight = 64

def alignFace(imFace, landmarks):
  l_x = landmarks[39][0]
  l_y = landmarks[39][1]
  r_x = landmarks[42][0]
  r_y = landmarks[42][1]
  dy = r_y - l_y
  dx = r_x - l_x
  angle = math.atan2(dy, dx) * 180.0 / math.pi  # Convert from radians to degrees

  eyesCenter = ((l_x + r_x)*0.5, (l_y + r_y)*0.5)
  rotMatrix = cv2.getRotationMatrix2D(eyesCenter, angle, 1)
  alignedImFace = np.zeros(imFace.shape, dtype=np.uint8)
  alignedImFace = cv2.warpAffine(imFace, rotMatrix, (imFace.shape[1],imFace.shape[0]))

  return alignedImFace

if __name__ == '__main__':

  # Now let's prepare our training data
  imagesFolder = '../data/images/FaceRec/trainFaces'
  subfolders = []
  for x in os.listdir(imagesFolder):
    xpath = os.path.join(imagesFolder, x)
    if os.path.isdir(xpath):
      subfolders.append(xpath)

  imagePaths = []
  labels = []

  labelsMap = {}
  labelsMap[-1] = "unknown"

  for i, subfolder in enumerate(subfolders):
    labelsMap[i] = os.path.basename(subfolder)
    for x in os.listdir(subfolder):
      xpath = os.path.join(subfolder, x)
      if x.endswith('jpg') or x.endswith('pgm'):
        imagePaths.append(xpath)
        labels.append(i)

  imagesFaceTrain = []
  labelsFaceTrain = []

  # Load face detector
  faceDetector = dlib.get_frontal_face_detector()

  # Load landmark detector.
  landmarkDetector = dlib.shape_predictor("../data/models/shape_predictor_68_face_landmarks.dat")

  for j, imagePath in enumerate(imagePaths):
    im = cv2.imread(imagePath, 0)
    imHeight, imWidth = im.shape[:2]
    # Detect faces in the image

    # Find landmarks.
    landmarks = fbc.getLandmarks(faceDetector, landmarkDetector, im)
    landmarks = np.array(landmarks)
    if len(landmarks) < 68:
      print("{}, Only {} Landmarks located".format(imagePath,len(landmarks)))
      continue
    else:
      print("Processing : {}".format(imagePath))

    x1Limit = landmarks[0][0] - (landmarks[36][0] - landmarks[0][0])
    x2Limit = landmarks[16][0] + (landmarks[16][0] - landmarks[45][0])
    y1Limit = landmarks[27][1] - 3*(landmarks[30][1] - landmarks[27][1])
    y2Limit = landmarks[8][1] + (landmarks[30][1] - landmarks[29][1])

    x1 = max(x1Limit,0)
    x2 = min(x2Limit, imWidth)
    y1 = max(y1Limit, 0)
    y2 = min(y2Limit, imHeight)
    imFace = im[y1:y2, x1:x2]

    alignedFace = alignFace(imFace, landmarks)
    alignedFace = cv2.resize(alignedFace, (faceHeight, faceWidth))

    imagesFaceTrain.append(np.float32(alignedFace)/255.0)
    labelsFaceTrain.append(labels[j])
    # fileParts = imagePath.split('/')
    # croppedFilename = "../../data/images/alignedFaces/{}_{}".format(fileParts[-2], fileParts[-1])
    # cv2.imwrite(croppedFilename, alignedFace)
    # k=cv2.waitKey(0)
    # if k == 27:
    #   cv2.destroyAllWindows()
    #   sys.exit()

  # Now we will train our model using EigenFace recognizer.

  faceRecognizerEigen = cv2.face.EigenFaceRecognizer_create()
  print("Training using Eigen Faces")
  faceRecognizerEigen.train(imagesFaceTrain, np.array(labelsFaceTrain))
  faceRecognizerEigen.write('face_model_eigen.yml')

  faceRecognizerFisher = cv2.face.FisherFaceRecognizer_create()
  print("Training using Fisher Faces")
  faceRecognizerFisher.train(imagesFaceTrain, np.array(labelsFaceTrain))
  faceRecognizerFisher.write('face_model_fisher.yml')

  faceRecognizerLBPH = cv2.face.LBPHFaceRecognizer_create()
  print("Training using LBP Histograms")
  faceRecognizerLBPH.train(imagesFaceTrain, np.array(labelsFaceTrain))
  faceRecognizerLBPH.write('face_model_lbph.yml')

  print("All Models saved")

  # save label number to label names mapping
  with open('labels_map.pkl', 'wb') as f:
    cPickle.dump(labelsMap, f)
