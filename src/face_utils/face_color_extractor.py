import dlib
import os
import numpy as np
from skimage.io import imread, imsave
from skimage.transform import rescale, resize
import cv2 as cv
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

class FaceColorExtractor():
    def __init__(self, dlib_model_path):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(dlib_model_path)


    def extract(self, face_img):
        img_kpts = self._detect_face_landmark(face_img)

        convex_idxs = [i for i in range(17)] + [26,25,24] + [19,18,17]
        polygon_pnts = img_kpts[convex_idxs, :].reshape(1,-1,2)
        polygon_pnts = polygon_pnts.astype(np.int32)

        mask = np.zeros(face_img.shape[:2], np.uint8)
        cv.fillConvexPoly(mask, polygon_pnts, 1)
        mask = mask.astype(np.bool)
        mask_nonface = np.bitwise_not(mask)

        n_classes = 15
        colors = face_img[mask,:]
        cov_type = 'full'
        gmm = GaussianMixture(n_components=n_classes, covariance_type=cov_type, max_iter=20, random_state=0)
        gmm.fit(colors)

        col_mask = np.sum(mask, axis=0)
        col_mask = np.argwhere(col_mask)
        x0 = max(np.min(col_mask) - 1, 0)
        x1 = np.max(col_mask) + 1

        row_mask = np.sum(mask, axis=1)
        row_mask = np.argwhere(row_mask)
        y0 = max(np.min(row_mask) - 1, 0)
        y1 = np.max(row_mask) + 1

        out_imgs = []
        sorted_idxs = np.argsort(gmm.weights_)
        for i in sorted_idxs:
            cls_color = gmm.means_[i, :].astype(np.uint8)
            print(gmm.weights_[i])
            img_1 = face_img.copy()
            img_1[mask_nonface] = cls_color
            d = 150
            img_1 = img_1[y0-d:y1+d, x0-d:x1+d, :]
            out_imgs.append(img_1)

        return out_imgs

    def _shape_to_np(self, shape, dtype="int"):
        # initialize the list of (x, y)-coordinates
        coords = np.zeros((shape.num_parts, 2), dtype=dtype)

        # loop over all facial landmarks and convert them
        # to a 2-tuple of (x, y)-coordinates
        for i in range(0, shape.num_parts):
            coords[i] = (shape.part(i).x, shape.part(i).y)

        # return the list of (x, y)-coordinates
        return coords

    def _detect_face_landmark(self, img):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # detect faces in the grayscale image
        rects = self.detector(gray, 1)
        # loop over the face detections
        for (i, rect) in enumerate(rects):
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = self.predictor(gray, rect)
            shape = self._shape_to_np(shape)

            shape = shape.astype(np.float32)
            shape = shape.reshape((-1, 2))
            return shape

        return None


if __name__ == '__main__':
    dlib_model_path = '/home/khanhhh/data_1/projects/Oh/codes/human_estimation/data/shape_predictor_68_face_landmarks.dat'
    extractor = FaceColorExtractor(dlib_model_path)

    img_path = '/home/khanhhh/data_1/projects/Oh/data/face/2019-06-04-face/MVIMG_20190604_180645.jpg'
    img = cv.imread(img_path)
    max_size = max(img.shape[0], img.shape[1])
    if max_size > 1000:
        img = rescale(img, 1000. / max_size)
        img = (img * 255).astype(np.uint8)

    out_dir = '/home/khanhhh/data_1/projects/Oh/data/face/face_skin_color'
    imgs = extractor.extract(img)

    for i, color_img in enumerate(imgs):
        cv.imwrite(filename=f'{out_dir}/cluster_color_{i}.jpg', img=color_img)