# Let's import necessary modules
import os,sys,dlib,glob
import cv2
import math
import numpy as np
import faceBlendCommon as fbc

try:
  import cPickle  # Python 2
except ImportError:
  import _pickle as cPickle  # Python 3

faceWidth = 64
faceHeight = 64
VIDEO = 0

MODEL = 'l'

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

  if VIDEO:
    cam = cv2.VideoCapture("../data/videos/face1.mp4")
  else:
    testFiles = glob.glob("../data/images/FaceRec/testFaces/*.jpg")
    # testFiles += glob.glob("../../data/images/testFaces/*.pgm")
    testFiles.sort()
    i = 0
    correct = 0
    error = 0


  if MODEL == 'e':
    # Create a face recognizer object
    faceRecognizer = cv2.face.EigenFaceRecognizer_create()
    # load model from disk
    faceRecognizer.read('face_model_eigen.yml')
  elif MODEL == 'f':
    # Create a face recognizer object
    faceRecognizer = cv2.face.FisherFaceRecognizer_create()
    # load model from disk
    faceRecognizer.read('face_model_fisher.yml')
  elif MODEL == 'l':
    # Create a face recognizer object
    faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
    # load model from disk
    faceRecognizer.read('face_model_lbph.yml')

  # load label numbers to label names mapping
  labelsMap = np.load('labels_map.pkl')

  # Load face detector
  faceDetector = dlib.get_frontal_face_detector()

  # Load landmark detector.
  landmarkDetector = dlib.shape_predictor("../data/models/shape_predictor_68_face_landmarks.dat")

  while VIDEO or (i < len(testFiles)):

    if VIDEO:
      success, original = cam.read()
      original = cv2.resize(original, (640, 480))
      if not success:
        print('cannot capture input from camera')
        break

    else:
      imagePath = testFiles[i]
      original = cv2.imread(imagePath)
      i += 1

    im = cv2.cvtColor(original,cv2.COLOR_BGR2GRAY)

    imHeight, imWidth = im.shape[:2]

    # Find landmarks.
    landmarks = fbc.getLandmarks(faceDetector, landmarkDetector, im)
    landmarks = np.array(landmarks)
    if len(landmarks) < 68:
      print("Only {} Landmarks located".format(len(landmarks)))
      continue

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
    imFaceFloat = np.float32(alignedFace)/255.0

    predictedLabel = -1
    predictedLabel, score = faceRecognizer.predict(imFaceFloat)
    center = ( int((x1 + x2) /2), int((y1 + y2)/2) )
    radius = int((y2-y1)/2.0)
    cv2.circle(original, center, radius, (0, 255, 0), 1, lineType=cv2.LINE_AA);
    cv2.putText(original, '{} {}'.format(labelsMap[predictedLabel],score), (10,100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2);

    cv2.imshow('Face Recognition Demo', original)
    if VIDEO:
      k = cv2.waitKey(10) & 0xff
    else:
      k = cv2.waitKey(300) & 0xff
    if k == 27:
      cv2.destroyAllWindows()
      if VIDEO:
        cam.release()
      break
