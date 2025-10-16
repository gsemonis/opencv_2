import cv2
import dlib
import numpy as np
from pathlib import Path


MODEL_PATH = r"C:\Users\Gabriel\OpenCv\Courses\opencv_2\project1-virtual-makeup\shape_predictor_68_face_landmarks.dat"
IMG_PATH   = r"C:\Users\Gabriel\OpenCv\Courses\opencv_2\project1-virtual-makeup\me.jpg"
ORIGINAL_IMAGE_WINDOW = "original"
AFTER_WINDOW = "after"

def main():
    # #load the image, create a mask mat
    # original = cv2.imread(IMG_PATH)
    # if original is None:
    #     raise FileNotFoundError(f"Could not read image at: {IMG_PATH}")
    # img_dlib = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    # img = original.copy()
    # height, width = img.shape[:2]
    scaler = .5

    #load dlib face and landmark detectors
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Missing dlib model at: {MODEL_PATH}")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(MODEL_PATH)

    #get the faces
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cv2.namedWindow(ORIGINAL_IMAGE_WINDOW, cv2.WINDOW_NORMAL)
        cv2.namedWindow(AFTER_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(ORIGINAL_IMAGE_WINDOW, np.int32(width / scaler), np.int32(height / scaler))
        cv2.resizeWindow(AFTER_WINDOW, np.int32(width / scaler), np.int32(height / scaler))
        while True:
            ret, img = cap.read()
            img_dlib = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = detector(img_dlib, 0)
            if faces is not None and len(faces) > 0:
                # get the first face
                face = faces[0]
                # get the landmarks
                shape = predictor(img_dlib, face)
                pts = np.array([(p.x, p.y) for p in shape.parts()], dtype=np.int32)

                # start making a mask of the lips
                lips_mask = np.zeros((height, width), dtype=np.uint8)
                outer_lips = pts[48:60]
                inner_lips = pts[60:]
                inner_lips_polygon = inner_lips.reshape((-1, 1, 2))
                outer_lips_polygon = outer_lips.reshape(-1, 1, 2)
                cv2.fillPoly(lips_mask, [outer_lips_polygon], (255, 255, 255))
                # dilate the outer edge a bit just because
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                lips_mask = cv2.dilate(lips_mask, kernel, iterations=1)
                cv2.fillPoly(lips_mask, [inner_lips_polygon], (0, 0, 0))
                mask = lips_mask > 0

                # convert image to hsv to manipulate the color
                image_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                hue, sat, val = cv2.split(image_hsv)
                boundary_saturation = 220
                max_saturation = sat[mask].max()
                if boundary_saturation >= max_saturation:
                    delta_saturation = boundary_saturation - max_saturation
                    sat[mask] += delta_saturation

                # put hsv back together and in bgr format
                image_hsv = cv2.merge([hue, sat, val])
                after = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)
                cv2.imshow(ORIGINAL_IMAGE_WINDOW, img)
                cv2.imshow(AFTER_WINDOW, after)
                if cv2.waitKey(5) & 0xFF == 27:
                    break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
