import cv2
import dlib
import numpy as np
from pathlib import Path

MODEL_PATH = r"C:\Users\Gabriel\OpenCv\Courses\opencv_2\project1-virtual-makeup\shape_predictor_68_face_landmarks.dat"
IMG_PATH   = r"C:\Users\Gabriel\OpenCv\Courses\opencv_2\project1-virtual-makeup\me.jpg"
ORIGINAL_IMAGE_WINDOW = "original"
AFTER_WINDOW = "after"

def scale_and_clip_rect(x, y, w, h, scale, max_width, max_height):
    cx = x + w / 2.0
    cy = y + h / 2.0
    # scale size
    new_w = w * scale
    new_h = h * scale
    # compute new top-left
    new_x = int(round(cx - new_w / 2.0))
    new_y = int(round(cy - new_h / 2.0))
    # clip to bounds
    new_x = max(0, new_x)
    new_y = max(0, new_y)
    new_w = int(round(new_w))
    new_h = int(round(new_h))
    if new_x + new_w > max_width:
        new_w = max_width - new_x
    if new_y + new_h > max_height:
        new_h = max_height - new_y
    return new_x, new_y, new_w, new_h

def create_iris_mask(image):
    img_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_grey = clahe.apply(img_grey)
    img_grey = cv2.medianBlur(img_grey, 5)
    circles = cv2.HoughCircles(img_grey, cv2.HOUGH_GRADIENT, dp=1.2, minDist=10, param1=80,param2=15,minRadius=10,maxRadius=100)
    if circles is not None:
        circles = np.uint16(np.around(circles[0]))
        h,w = img_grey.shape
        valid = [c for c in circles if 5 < c[0] < w - 5 and 5 < c[1] < h - 5]
        if valid:
            mid_y = h // 2
            mid_x = w // 2
            valid = sorted(valid, key=lambda c: ((c[0] - mid_x) ** 2 + (c[1] - mid_y) ** 2, c[2]))
            pupil = valid[0]
            iris = max(valid, key=lambda c: c[2])
            cv2.circle(img_grey, (pupil[0], pupil[1]), pupil[2], 255, 2)
            cv2.circle(img_grey, (iris[0], iris[1]), iris[2], 255, 2)

    return img_grey

def main():
    original = cv2.imread(IMG_PATH,cv2.IMREAD_UNCHANGED)
    if original is None:
        raise FileNotFoundError(f"Could not read image at: {IMG_PATH}")

    img_dlib = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    height, width = img_dlib.shape[:2]
    scaler = .5
    cv2.namedWindow(ORIGINAL_IMAGE_WINDOW, cv2.WINDOW_NORMAL)
    cv2.namedWindow(AFTER_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(ORIGINAL_IMAGE_WINDOW, np.int32(width * scaler), np.int32(height * scaler))
    cv2.resizeWindow(AFTER_WINDOW, np.int32(width * scaler), np.int32(height * scaler))
    # load dlib face and landmark detectors
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Missing dlib model at: {MODEL_PATH}")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(MODEL_PATH)
    faces = detector(img_dlib, 0)
    if faces is not None and len(faces) > 0:
        face = faces[0]
        shape = predictor(img_dlib, face)
        points = np.array([(p.x, p.y) for p in shape.parts()], dtype=np.int32)

        img = img_dlib.copy()
        height, width = img.shape[:2]
        right_eye_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        left_eye_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        right_eye_polygon = np.array(points[36:42], dtype=np.int32)
        right_eye_polygon = right_eye_polygon.reshape(-1, 1, 2)
        left_eye_polygon = np.array(points[42:48], dtype=np.int32)
        left_eye_polygon = left_eye_polygon.reshape(-1, 1, 2)

        x, y, w, h = cv2.boundingRect(right_eye_polygon)
        x, y, w, h = scale_and_clip_rect(x, y, w, h, 1.4, width, height)
        cv2.rectangle(right_eye_mask, (x, y), (x + w, y + h), 255, -1)
        right_eye = cv2.bitwise_and(img, img, mask=right_eye_mask)
        right_iris_mask = create_iris_mask(right_eye[y : y + h, x : x + w,:])

        x, y, w, h = cv2.boundingRect(left_eye_polygon)
        x, y, w, h = scale_and_clip_rect(x, y, w, h, 1.4, width, height)
        cv2.rectangle(left_eye_mask, (x, y), (x + w, y + h), 255, -1)
        left_eye = cv2.bitwise_and(img, img, mask=left_eye_mask)
        #left_iris_mask = create_iris_mask(left_eye[y : y + h, x : x + w,:])

        cv2.imshow(ORIGINAL_IMAGE_WINDOW, img[:,:,::-1])
        cv2.imshow("asdf", right_iris_mask)
        #cv2.imshow(AFTER_WINDOW, right_iris_mask)
        while True:
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()