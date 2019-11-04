import dlib
import cv2 as cv
import face_utils
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os

detector = dlib.get_frontal_face_detector()
shape_predictor_path = '/home/khanhhh/data_1/projects/Oh/codes/human_estimation/data/shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(shape_predictor_path)

def _detect_face_landmark(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        shape = shape.astype(np.float32)
        shape = shape.reshape((1, -1, 2))
        return shape


    return None

import time
if __name__ == "__main__":
    dir = "/home/khanhhh/data_1/projects/Oh/data/face/google_front_faces/"
    out_dir = "/home/khanhhh/data_1/projects/Oh/data/face/google_front_faces/result_landmarks/"
    os.makedirs(out_dir, exist_ok = True)
    paths = [path for path in Path(dir).glob("*.jpg")]
    for path in tqdm(paths):
        print(path)
        img = cv.imread(str(path))

        start = time.time()
        points  = _detect_face_landmark(img)
        end = time.time()
        print(end - start)
        if points is not None:
            points = points.astype(np.int32)
            for i in range(points.shape[1]):
                p = points[0,i,:]
                cv.circle(img, (p[0], p[1]), 2, (0,0,255), thickness=2, lineType=cv.FILLED)

        cv.imwrite(f'{out_dir}/{path.name}', img)
