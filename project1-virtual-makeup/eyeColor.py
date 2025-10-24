from pickletools import uint8

import cv2
import dlib
import numpy as np
from pathlib import Path

MODEL_PATH = r"C:\Users\Gabriel\OpenCv\Courses\opencv_2\project1-virtual-makeup\shape_predictor_68_face_landmarks.dat"
IMG_PATH   = r"C:\Users\Gabriel\OpenCv\Courses\opencv_2\project1-virtual-makeup\me.jpg"
ORIGINAL_IMAGE_WINDOW = "original"
AFTER_WINDOW = "after"

def drawLandmarkPoints(image, points):
    for i, (x,y) in enumerate(points):
        cv2.circle(image, (int(x), int(y)), 20, (0,0,255), -1, lineType=cv2.LINE_AA)
        cv2.putText(image, str(i), (int(x) + 3, int(y) - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (255,0,255), 3, cv2.LINE_AA)
def alphaBlend(alpha, foreground, background):
    a = (alpha.astype(np.float32) / 255.0)
    a = cv2.merge([a,a,a])
    fg = foreground.astype(np.float32)
    bg = background.astype(np.float32)
    fg = a * fg
    bg = (1 - a) * bg
    output = cv2.add(fg, bg)
    return output.astype(np.uint8)
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
        npoints = np.array([(p.x, p.y) for p in shape.parts()], dtype=np.int32)

        blush_color_rgb = (255, 102, 102)
        blush_color_rgb = (0, 0, 255)
        kernel_size = 543
        blush_line_width = int(round(np.linalg.norm(npoints[1] - npoints[2])) // 2)

        img_rgb = img_dlib.copy()
        height, width = img_rgb.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        hull = cv2.convexHull(npoints)
        cv2.fillConvexPoly(mask, hull, 255)
        blush_lines = np.zeros((height, width), dtype=np.uint8)
        blush_image = np.ones_like(img_rgb, np.uint8) * blush_color_rgb

        cv2.line(blush_lines, npoints[2], npoints[30], 255, blush_line_width, cv2.LINE_AA)
        cv2.line(blush_lines, npoints[14], npoints[30], 255, blush_line_width, cv2.LINE_AA)

        blush_lines = cv2.resize(blush_lines, (width // 2, height // 2))
        blush_lines = cv2.GaussianBlur(blush_lines, (kernel_size, kernel_size), 0)
        blush_lines = cv2.resize(blush_lines, (width, height))
        blush_lines = cv2.bitwise_and(blush_lines, mask)

        img_rgb = alphaBlend(blush_lines, blush_image, img_rgb)

        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        cv2.imshow(ORIGINAL_IMAGE_WINDOW, img_bgr)
        cv2.imshow(AFTER_WINDOW, blush_lines)
        while True:
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()