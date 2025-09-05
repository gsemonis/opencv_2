import sys
import cv2
import dlib
import math
import numpy as np
from renderFace import renderFace3
from dataPath import DATA_PATH

PREDICTOR_PATH = DATA_PATH + "models/shape_predictor_68_face_landmarks.dat"
RESIZE_HEIGHT = 320
NUM_FRAMES_FOR_FPS = 100
SKIP_FRAMES = 1


# 3D Model Points of selected landmarks in an arbitrary frame of reference
def get3dModelPoints():
  Model3D=np.array([[-7.308957,0.913869,0.000000], [-6.775290,-0.730814,-0.012799], [-5.665918,-3.286078,1.022951], [-5.011779,-4.876396,1.047961], [-4.056931,-5.947019,1.636229],   [-1.833492,-7.056977,4.061275], [0.000000,-7.415691,4.070434], [1.833492,-7.056977,4.061275], [4.056931,-5.947019,1.636229], [5.011779,-4.876396,1.047961], [5.665918,-3.286078,1.022951], [6.775290,-0.730814,-0.012799], [7.308957,0.913869,0.000000], [5.311432,5.485328,3.987654], [4.461908,6.189018,5.594410], [3.550622,6.185143,5.712299], [2.542231,5.862829,4.687939], [1.789930,5.393625,4.413414], [2.693583,5.018237,5.072837], [3.530191,4.981603,4.937805], [4.490323,5.186498,4.694397], [-5.311432,5.485328,3.987654], [-4.461908,6.189018,5.594410], [-3.550622,6.185143,5.712299], [-2.542231,5.862829,4.687939], [-1.789930,5.393625,4.413414], [-2.693583,5.018237,5.072837], [-3.530191,4.981603,4.937805], [-4.490323,5.186498,4.694397], [1.330353,7.122144,6.903745], [2.533424,7.878085,7.451034], [4.861131,7.878672,6.601275], [6.137002,7.271266,5.200823], [6.825897,6.760612,4.402142], [-1.330353,7.122144,6.903745], [-2.533424,7.878085,7.451034], [-4.861131,7.878672,6.601275], [-6.137002,7.271266,5.200823], [-6.825897,6.760612,4.402142], [-2.774015,-2.080775,5.048531], [-0.509714,-1.571179,6.566167], [0.000000,-1.646444,6.704956], [0.509714,-1.571179,6.566167], [2.774015,-2.080775,5.048531], [0.589441,-2.958597,6.109526], [0.000000,-3.116408,6.097667], [-0.589441,-2.958597,6.109526], [-0.981972,4.554081,6.301271], [-0.973987,1.916389,7.654050], [-2.005628,1.409845,6.165652], [-1.930245,0.424351,5.914376], [-0.746313,0.348381,6.263227], [0.000000,1.400000,8.063430], [0.746313,0.348381,6.263227], [1.930245,0.424351,5.914376], [2.005628,1.409845,6.165652], [0.973987,1.916389,7.654050], [0.981972,4.554081,6.301271]]);
  alpha=-1
  face_3d_points = np.array([[ Model3D[13][0], Model3D[13][1]  , -alpha*(Model3D[13][2]- Model3D[52][2])],
            [ Model3D[17][0], Model3D[17][1]  , -alpha*(Model3D[17][2]-Model3D[52][2])],
            [ Model3D[25][0], Model3D[25][1]  , -alpha*(Model3D[25][2]-Model3D[52][2])],
            [ Model3D[21][0], Model3D[21][1]  , -alpha*(Model3D[21][2]-Model3D[52][2])],
            [ Model3D[43][0], Model3D[43][1]  , -alpha*(Model3D[43][2]-Model3D[52][2])],
            [ Model3D[39][0], Model3D[39][1]  , -alpha*(Model3D[39][2]-Model3D[52][2])],
            [ Model3D[52][0], Model3D[52][1]  , -alpha*(Model3D[52][2]-Model3D[52][2])]],dtype="double")
  #print(face_3d_points)
  return face_3d_points


# 2D landmark points from all landmarks
def get2dImagePoints(shape):
  imagePoints = [[shape[36][0], shape[36][1]],
                 [shape[39][0], shape[39][1]],
                 [shape[42][0], shape[42][1]],
                 [shape[45][0], shape[45][1]],
                 [shape[48][0], shape[48][1]],
                 [shape[54][0], shape[54][1]],
                 [shape[30][0], shape[30][1]]]
  #print(imagePoints)
  return np.array(imagePoints, dtype=np.float64)


# Camera Matrix from focal length and focal center
def getCameraMatrix(focalLength, center):
  focalLength=2*focalLength
  #print(focalLength)
  cameraMatrix = [[focalLength, 0, center[0]],
                  [0, focalLength, center[1]],
                  [0, 0, 1]]
  return np.array(cameraMatrix, dtype=np.float64)



# Function to calculate the intereye distance.
def interEyeDistance(predict):
  leftEyeLeftCorner = (predict[36].x, predict[36].y)
  rightEyeRightCorner = (predict[45].x, predict[45].y)
  distance = cv2.norm(np.array(rightEyeRightCorner) - np.array(leftEyeLeftCorner))
  distance = int(distance)
  return distance


try:
    # Initializing video capture object.
    #cap = cv2.VideoCapture('/home/pranav/datasets/landmark-datasets/300VW_Dataset_2015_12_14/001/vid.avi')
    cap = cv2.VideoCapture(0)
    print("Initializing capture")
    # Error message if camera fails to open.
    if(cap.isOpened()==False):
      print("Unable to connect to camera")


    winSize = 101
    maxLevel = 10
    # fps = 30.0
    # Grab a frame
    ret,imPrev = cap.read()

    # Convert to grayscale.
    imGrayPrev = cv2.cvtColor(imPrev, cv2.COLOR_BGR2GRAY)

    # Finding the size of the image.
    size = imPrev.shape[0:1]

    print("Loading dlib")
    detector = dlib.get_frontal_face_detector()
    print("Loading shape file")
    landmarkDetector = dlib.shape_predictor(PREDICTOR_PATH)

    # Initializing the parameters
    points=[]
    pointsPrev=[]
    pointsDetectedCur=[]
    pointsDetectedPrev=[]

    eyeDistanceNotCalculated = True
    eyeDistance = 0
    isFirstFrame = True
    # Initial value, actual value calculated after 100 frames
    fps = 25
    showStabilized = True
    count =0
    frame_no = 1
    print("Starting processing")
    while(True):
      if (count==0):
        t = cv2.getTickCount()

      # Grab a frame
      ret,im = cap.read()
      if ret==False:
        break
      # Converting to grayscale
      imGray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
      height = im.shape[0]
      IMAGE_RESIZE = float(height)/RESIZE_HEIGHT
      # Resize image for faster face detection
      imSmall = cv2.resize(im, None, fx=1.0/IMAGE_RESIZE, fy=1.0/IMAGE_RESIZE,interpolation = cv2.INTER_LINEAR)

      # Skipping the frames for faster processing
      if (count % SKIP_FRAMES == 0):
        faces = detector(cv2.cvtColor(imSmall, cv2.COLOR_BGR2RGB),0)

      # If no face was detected
      if len(faces)==0:
        pass

      # If faces are detected, iterate through each image and detect landmark points
      else:
        # get 3D model points
        modelPoints = get3dModelPoints()

        for i in range(0,len(faces)):
          #print("face detected")
          # Face detector was found over a smaller image.
          # So, we scale face rectangle to correct size.
          newRect = dlib.rectangle(int(faces[i].left() * IMAGE_RESIZE),
            int(faces[i].top() * IMAGE_RESIZE),
            int(faces[i].right() * IMAGE_RESIZE),
            int(faces[i].bottom() * IMAGE_RESIZE))
          #print(newRect)
          # Handling the first frame of video differently,for the first frame copy the current frame points
          if (isFirstFrame==True):
            pointsPrev=[]
            pointsDetectedPrev = []
            [pointsPrev.append((p.x, p.y)) for p in landmarkDetector(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), newRect).parts()]
            [pointsDetectedPrev.append((p.x, p.y)) for p in landmarkDetector(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), newRect).parts()]
            #print("NON-PARTS")
            #print(landmarkDetector(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), newRect))
            #print("PARTS")
            #print(landmarkDetector(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), newRect).parts())

          # If not the first frame, copy points from previous frame.
          else:
            pointsPrev=[]
            pointsDetectedPrev = []
            pointsPrev = points
            pointsDetectedPrev = pointsDetectedCur

          # pointsDetectedCur stores results returned by the facial landmark detector
          # points stores the stabilized landmark points
          points = []
          pointsDetectedCur = []
          [points.append((p.x, p.y)) for p in landmarkDetector(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), newRect).parts()]
          #print(landmarkDetector(cv2.cvtColor(im,cv2.COLOR_BGR2RGB),newRect))
          #print(landmarkDetector(cv2.cvtColor(im,cv2.COLOR_BGR2RGB),newRect).parts())
          [pointsDetectedCur.append((p.x, p.y)) for p in landmarkDetector(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), newRect).parts()]

          # Convert to numpy float array
          pointsArr = np.array(points,np.float32)
          pointsPrevArr = np.array(pointsPrev,np.float32)

          # If eye distance is not calculated before
          if eyeDistanceNotCalculated:
            eyeDistance = interEyeDistance(landmarkDetector(cv2.cvtColor(im, cv2.COLOR_BGR2RGB),newRect).parts())
            #print(eyeDistance)
            eyeDistanceNotCalculated = False

          if eyeDistance > 100:
              dotRadius = 3
          else:
            dotRadius = 2

          #print(eyeDistance)
          sigma = eyeDistance * eyeDistance / 400
          s = 2*int(eyeDistance/4)+1

          #  Set up optical flow params
          lk_params = dict(winSize  = (s, s), maxLevel = 5, criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 20, 0.03))
          # Python Bug. Calculating pyramids and then calculating optical flow results in an error. So directly images are used.
          # ret, imGrayPyr= cv2.buildOpticalFlowPyramid(imGray, (winSize,winSize), maxLevel)

          pointsArr,status, err = cv2.calcOpticalFlowPyrLK(imGrayPrev,imGray,pointsPrevArr,pointsArr,**lk_params)
          sigma =100

          # Converting to float
          pointsArrFloat = np.array(pointsArr,np.float32)

          # Converting back to list
          points = pointsArrFloat.tolist()

          # Final landmark points are a weighted average of
          # detected landmarks and tracked landmarks
          for k in range(0,len(landmarkDetector(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), newRect).parts())):
            d = cv2.norm(np.array(pointsDetectedPrev[k]) - np.array(pointsDetectedCur[k]))
            alpha = math.exp(-d*d/sigma)
            alpha=0
            points[k] = (1 - alpha) * np.array(pointsDetectedCur[k]) + alpha * np.array(points[k])

          # # Drawing over the stabilized landmark points
          # if showStabilized is True:
          for p in points:
               cv2.circle(im,(int(p[0]),int(p[1])),dotRadius, (255,0,0),-1)
          # else:
          #   for p in pointsDetectedCur:
          #     cv2.circle(im,(int(p[0]),int(p[1])),dotRadius, (0,0,255),-1)
                # Draw landmarks over face
          renderFace3(im, points)

          # get 2D landmarks from Dlib's shape object
          imagePoints = get2dImagePoints(points)

          # Camera parameters
          rows, cols, ch = im.shape
          focalLength = cols/2
          cameraMatrix = getCameraMatrix(focalLength, (rows/2, cols/2))

          #initialize rotation and translation vector to a value.
          rotationVector = np.array([[0,0,0.1]],dtype="double")
          translationVector=np.array([[0,0,100.0]],dtype="double")

          #initialize distortion coeffs to 0.
          distCoeffs = np.array([[ 0,   0,   0,   0,
             0]],dtype= "double") # Assuming no lens distortion

          #Use solvePnP to get Rotation and translation vectors. The algorithm used here will minimize the error between the projected face_3d_points on a plane using the rotation and translation vectors and face_2d_points.
          (success, rotationVector, translationVector) = cv2.solvePnP(modelPoints, imagePoints, cameraMatrix, distCoeffs)

          # # Assume no lens distortion
          # distCoeffs = np.zeros((4, 1), dtype=np.float64)

          # # calculate rotation and translation vector using solvePnP
          # success, rotationVector, translationVector = cv2.solvePnP(modelPoints, imagePoints, cameraMatrix, distCoeffs)

          # Project a 3D point (0, 0, 1000.0) onto the image plane.
          # We use this to draw a line sticking out of the nose
          noseEndPoints3D = np.array([[0, 0, 20.0]], dtype=np.float64)
          noseEndPoint2D, jacobian = cv2.projectPoints(noseEndPoints3D, rotationVector, translationVector, cameraMatrix, distCoeffs)

          # points to draw line
          p1 = (int(imagePoints[6, 0]), int(imagePoints[6, 1]))
          p2 = (int(noseEndPoint2D[0, 0, 0]), int(noseEndPoint2D[0, 0, 1]))

          # draw line using points P1 and P2
          cv2.line(im, p1, p2, (110, 220, 0), thickness=2, lineType=cv2.LINE_AA)
          # Print actual FPS
          #cv2.putText(im, "fps: {}".format(fps), (50, size[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

          # Resize image for display
          imDisplay = cv2.resize(im, None, fx=0.5, fy=0.5)
          # cv2.imwrite("/Users/sowjanyakonduri/Downloads/00001_pose/" + str(frame_no).zfill(4) + '.jpg', imDisplay)

          # WaitKey slows down the runtime quite a lot
          # So check every 15 frames
          #if (count % 15 == 0):
            #key = cv2.waitKey(1) & 0xFF

            # Stop the program.
            #if key==27:  # ESC
              # If ESC is pressed, exit.
              #sys.exit()

          isFirstFrame = False
          count = count+1

          # Calculating the fps value
          if ( count == NUM_FRAMES_FOR_FPS):
            t = (cv2.getTickCount()-t)/cv2.getTickFrequency()
            fps = NUM_FRAMES_FOR_FPS/t
            count = 0
            isFirstFrame = True

          # # Display the landmarks points
          cv2.putText(im, "{:0.2f}-fps".format(fps), (50, size[0]-50), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255), 3,cv2.LINE_AA)
          cv2.imshow("Frame", im)
          key = cv2.waitKey(25) & 0xFF
          # Stop the program
          if key==27:   # ESC
              # If ESC is pressed, exit.
              sys.exit()
          # # Use spacebar to toggle between Stabilized and Unstabilized version.
          # if key==32:
          #   showStabilized = not showStabilized

          # # Stop the program.
          # if key==27:
          #   sys.exit()
          # Getting ready for next frame
          imPrev = im
          imGrayPrev = imGray
        #frameName ="output_"+str(frame_no).zfill(5)+".png"
        #cv2.imwrite(frameName,im)
        frame_no = frame_no + 1
    cap.release()
    cv2.destroyAllWindows()

except Exception as e:
  #print(e)
  pass
