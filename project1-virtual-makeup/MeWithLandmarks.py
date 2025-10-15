import cv2
import dlib
import numpy as np
from pathlib import Path


MODEL_PATH = r"C:\Users\Gabriel\OpenCv\Courses\opencv_2\project1-virtual-makeup\shape_predictor_68_face_landmarks.dat"
IMG_PATH   = r"C:\Users\Gabriel\OpenCv\Courses\opencv_2\project1-virtual-makeup\me.jpg"
ORIGINAL_IMAGE_WINDOW = "original"
AFTER_WINDOW = "after"

def main():
    #load the image, create a mask mat
    original = cv2.imread(IMG_PATH)
    if original is None:
        raise FileNotFoundError(f"Could not read image at: {IMG_PATH}")
    img_dlib = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    img = original.copy()
    height, width = img.shape[:2]

    scaler = 3

    #load dlib face and landmark detectors
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Missing dlib model at: {MODEL_PATH}")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(MODEL_PATH)

    #get the faces
    faces = detector(img_dlib, 0)

    if not faces:
        raise ValueError("No faces detected in the image.")
    else:
        #get the first face
        face = faces[0]
        #get the landmarks
        shape = predictor(img_dlib, face)
        pts = np.array([(p.x, p.y) for p in shape.parts()], dtype=np.int32)
        #start creating masks and the image of the lips for cloning
        lips_image = np.zeros_like(img, dtype=np.uint8)
        lips_ring = lips_image[:,:,0].copy()

        outer_lips = pts[48:60]
        inner_lips = pts[60:]
        inner_lips_polygon = inner_lips.reshape((-1,1,2))
        outer_lips_polygon = outer_lips.reshape(-1,1,2)

        cv2.fillPoly(lips_ring, [outer_lips_polygon], (255, 255, 255))
        cv2.fillPoly(lips_image,[outer_lips_polygon], (255,255,255))
        cv2.fillPoly(lips_image, [inner_lips_polygon], (0, 0, 0))
        seamless_mask = lips_image[:, :, 0].copy()

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        lips_ring_dilated = cv2.dilate(lips_ring, kernel, iterations=1)
        lips_ring = lips_ring_dilated - lips_ring

        moments = cv2.moments(seamless_mask, binaryImage=True)
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])

        lips_image = cv2.bitwise_and(lips_image, img)
        #go to hsv to manipulate the color
        lips_image = cv2.cvtColor(lips_image, cv2.COLOR_BGR2HSV)

        hue,sat,val = cv2.split(lips_image)
        img[lips_ring] = 0
        hue[seamless_mask > 0] = 0
        sat[seamless_mask > 0] = 245
        # val[seamless_mask > 0] = 250

        lips_image = cv2.merge([hue,sat,val])
        lips_image = cv2.cvtColor(lips_image,cv2.COLOR_HSV2BGR)

        after = cv2.seamlessClone(lips_image, img, seamless_mask, p=(cx, cy), flags=cv2.MIXED_CLONE)

    cv2.namedWindow(ORIGINAL_IMAGE_WINDOW, cv2.WINDOW_NORMAL)
    cv2.namedWindow(AFTER_WINDOW, cv2.WINDOW_NORMAL)

    cv2.resizeWindow(ORIGINAL_IMAGE_WINDOW, np.int32(width / scaler), np.int32(height / scaler))
    cv2.resizeWindow(AFTER_WINDOW, np.int32(width / scaler), np.int32(height / scaler))
    cv2.imshow(ORIGINAL_IMAGE_WINDOW, img)
    cv2.imshow(AFTER_WINDOW, lip_ring)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
