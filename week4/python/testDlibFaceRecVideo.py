import os,sys,time
import dlib
import cv2
import numpy as np


# Path to landmarks and face recognition model files
PREDICTOR_PATH = '../data/models/shape_predictor_68_face_landmarks.dat'
FACE_RECOGNITION_MODEL_PATH = '../data/models/dlib_face_recognition_resnet_model_v1.dat'
SKIP_FRAMES = 1
THRESHOLD = 0.5

# Initialize face detector, facial landmarks detector and face recognizer
faceDetector = dlib.get_frontal_face_detector()
shapePredictor = dlib.shape_predictor(PREDICTOR_PATH)
faceRecognizer = dlib.face_recognition_model_v1(FACE_RECOGNITION_MODEL_PATH)

# load descriptors and index file generated during enrollment
index = np.load('index.pkl')
faceDescriptorsEnrolled = np.load('descriptors.npy')

# create a VideoCapture object
cam = cv2.VideoCapture("../data/videos/face1.mp4")

count = 0
# process frames from camera/video feed
while True:
  t = time.time()

  # capture frame
  success, im = cam.read()

  # exit if unable to read frame from camera/video feed
  if not success:
    print('cannot capture input from camera')
    break

  # We will be processing frames after an interval
  # of SKIP_FRAMES to increase processing speed
  if (count % SKIP_FRAMES) == 0:

    # convert image from BGR to RGB
    # because Dlib used RGB format
    img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    # detect faces in image
    faces = faceDetector(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Now process each face we found
    for face in faces:

      # Find facial landmarks for each detected face
      shape = shapePredictor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), face)

      # find coordinates of face rectangle
      x1 = face.left()
      y1 = face.top()
      x2 = face.right()
      y2 = face.bottom()

      # Compute face descriptor using neural network defined in Dlib
      # using facial landmark shape
      faceDescriptor = faceRecognizer.compute_face_descriptor(img, shape)

      # Convert face descriptor from Dlib's format to list, then a NumPy array
      faceDescriptorList = [m for m in faceDescriptor]
      faceDescriptorNdarray = np.asarray(faceDescriptorList, dtype=np.float64)
      faceDescriptorNdarray = faceDescriptorNdarray[np.newaxis, :]

      # Calculate Euclidean distances between face descriptor calculated on face dectected
      # in current frame with all the face descriptors we calculated while enrolling faces
      distances = np.linalg.norm(faceDescriptorsEnrolled - faceDescriptorNdarray, axis=1)
      # Calculate minimum distance and index of this face
      argmin = np.argmin(distances)  # index
      minDistance = distances[argmin]  # minimum distance

      # Dlib specifies that in general, if two face descriptor vectors have a Euclidean
      # distance between them less than 0.6 then they are from the same
      # person, otherwise they are from different people.

      # This threshold will vary depending upon number of images enrolled
      # and various variations (illuminaton, camera quality) between
      # enrolled images and query image
      # We are using a threshold of 0.5

      # If minimum distance if less than threshold
      # find the name of person from index
      # else the person in query image is unknown
      if minDistance <= THRESHOLD:
        label = index[argmin]
      else:
        label = 'unknown'

      print("time taken = {:.3f} seconds".format(time.time() - t))

      # Draw a rectangle for detected face
      cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255))

      # Draw circle for face recognition
      center = (int((x1 + x2)/2.0), int((y1 + y2)/2.0))
      radius = int((y2-y1)/2.0)
      color = (0, 255, 0)
      cv2.circle(im, center, radius, color, thickness=1, lineType=8, shift=0)

      # Write text on image specifying identified person and minimum distance
      org = (int(x1), int(y1))  # bottom left corner of text string
      font_face = cv2.FONT_HERSHEY_SIMPLEX
      font_scale = 0.8
      text_color = (255, 0, 0)
      printLabel = '{} {:0.4f}'.format(label, minDistance)
      cv2.putText(im, printLabel, org, font_face, font_scale, text_color, thickness=2)

    # Show result
    cv2.imshow('webcam', im)
  # Quit when Esc is pressed
  k = cv2.waitKey(1) & 0xff
  if k == 27:
    break  # esc pressed

  # Counter used for skipping frames
  count += 1
cv2.destroyAllWindows()
