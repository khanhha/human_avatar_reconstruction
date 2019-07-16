import dlib
import os
import numpy as np
from skimage.io import imread, imsave
from skimage.transform import rescale, resize
import cv2 as cv
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.cluster import SpectralClustering

class FaceColorExtractor():
    def __init__(self, dlib_model_path):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(dlib_model_path)

    def extract(self, face_img, sample_skin, hsv = False):
        debug_imgs, skin_color = self._extract_gmm(face_img, hsv)

        L = cv.Laplacian(cv.cvtColor(sample_skin, cv.COLOR_BGR2GRAY), cv.CV_32F, ksize=7)
        base_img = np.zeros(sample_skin.shape, dtype=np.uint8)
        print(base_img.shape)
        base_img[:,:,0] = skin_color[0]
        base_img[:,:,1] = skin_color[1]
        base_img[:,:,2] = skin_color[2]

        base_img_1 = base_img.copy()
        L = 0.3*(L /  L.max())
        base_img_1[:,:,0] = base_img_1[:,:,0] + base_img_1[:,:,0]*L
        base_img_1[:,:,1] = base_img_1[:,:,1] + base_img_1[:,:,1]*L
        base_img_1[:,:,2] = base_img_1[:,:,2] + base_img_1[:,:,2]*L

        f, axes = plt.subplots(2, 2, gridspec_kw={'wspace': 0, 'hspace': 0})
        for i, ax in enumerate(f.axes):
            ax.set_xticklabels([])
            ax.set_yticklabels([])

        axes[0,0].imshow(sample_skin[:,:,::-1])
        axes[0,1].imshow(L, cmap='gray')
        axes[1,0].imshow(base_img[:,:,::-1])
        axes[1,1].imshow(base_img_1[:,:,::-1])
        plt.show()
        return debug_imgs

    def _extract_spectral_clustering(self, face_img, hsv):
        img_kpts = self._detect_face_landmark(face_img)
        if hsv:
            face_img = cv.cvtColor(face_img, cv.COLOR_BGR2HSV)

        mask, mask_nonface = self._facial_masks(img_kpts, face_img.shape)

        n_classes = 5
        colors = face_img[mask, :]
        print(colors.shape)
        model = SpectralClustering(n_clusters=n_classes)
        cls_preds = model.fit_predict(colors)
        print(cls_preds)

    def _extract_gmm(self, face_img, hsv):
        img_kpts = self._detect_face_landmark(face_img)

        if hsv:
            face_img = cv.cvtColor(face_img, cv.COLOR_BGR2LAB)

        mask, mask_nonface = self._facial_masks(img_kpts, face_img.shape)

        # plt.imshow(face_img)
        # plt.imshow(mask)
        # plt.show()

        n_classes = 5
        colors = face_img[mask, :]
        cov_type = 'full'
        #gmm = GaussianMixture(n_components=n_classes, covariance_type=cov_type, max_iter=20, random_state=0)
        gmm = BayesianGaussianMixture(n_components=n_classes, covariance_type=cov_type, max_iter=20, random_state=0)
        cls_preds = gmm.fit_predict(colors)
        sorted_idxs = np.argsort(-gmm.weights_)
        best_idx = sorted_idxs[0]
        best_colors_mask = cls_preds == best_idx
        best_colors = colors[best_colors_mask]
        print(gmm.means_[best_idx, :])
        best_color = self._analysize(best_colors)

        x0,x1, y0, y1 = self._rect_face(mask)

        out_imgs = []
        for i in sorted_idxs:
            cls_color = gmm.means_[i, :].astype(np.uint8)
            print(gmm.weights_[i])
            img_1 = face_img.copy()
            img_1[mask_nonface] = cls_color
            d = 150
            img_1 = img_1[y0 - d:y1 + d, x0 - d:x1 + d, :]
            out_imgs.append(img_1)

        if hsv:
            for i in range(len(out_imgs)):
                img = out_imgs[i]
                img = cv.cvtColor(img, cv.COLOR_LAB2BGR)
                out_imgs[i] = img

        return out_imgs, best_color

    def _rect_face(self, mask):
        col_mask = np.sum(mask, axis=0)
        col_mask = np.argwhere(col_mask)
        x0 = max(np.min(col_mask) - 1, 0)
        x1 = np.max(col_mask) + 1

        row_mask = np.sum(mask, axis=1)
        row_mask = np.argwhere(row_mask)
        y0 = max(np.min(row_mask) - 1, 0)
        y1 = np.max(row_mask) + 1

        return x0, x1, y0, y1

    def _facial_masks(self, img_kpts, shape):
        convex_idxs = [i for i in range(17)] + [26, 25, 24] + [19, 18, 17]
        polygon_pnts = img_kpts[convex_idxs, :].reshape(1, -1, 2).astype(np.int32)

        mask_all_face = np.zeros(shape[:2], np.uint8)
        mask_all_face = cv.fillConvexPoly(mask_all_face, polygon_pnts, 1).astype(np.bool)
        mask_nonface = np.bitwise_not(mask_all_face)

        eye_idxs = [17, 18, 19] + [24, 25, 26] + [45, 46] + [40, 41, 36]
        mouth_idxs = [i for i in range(48, 59 + 1)]

        eye_mask = np.zeros(shape[:2], np.uint8)
        eye_mask = cv.fillConvexPoly(eye_mask, img_kpts[eye_idxs, :].reshape(1, -1, 2).astype(np.int32), 1).astype(
            np.bool)

        mouth_mask = np.zeros(shape[:2], np.uint8)
        mouth_mask = cv.fillConvexPoly(mouth_mask, img_kpts[mouth_idxs, :].reshape(1, -1, 2).astype(np.int32),
                                       1).astype(np.bool)

        mask_nonface = np.bitwise_or(np.bitwise_or(mask_nonface, mouth_mask), eye_mask)
        mask = np.bitwise_not(mask_nonface)

        return mask, mask_nonface

    def _analysize(self, colors):
        # import scipy.stats as stats
        # print(stats.describe(colors[:,0]))
        # print(np.mean(colors[:,0]))
        # print(stats.describe(colors[:,1]))
        # print(np.mean(colors[:,1]))
        # print(stats.describe(colors[:,2]))
        # print(np.mean(colors[:,2]))
        # for i in range(1,4):
        #     plt.subplot(1,3,i)
        #     plt.hist(colors[:,i-1], 254)
        # plt.show()
        return np.mean(colors, axis=0).astype(np.uint8)

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

    sample_skin_path = '/home/khanhhh/data_1/projects/Oh/data/face/sample_skin.jpg'
    sample_skin = cv.imread(sample_skin_path)

    img = cv.imread(img_path)
    max_size = max(img.shape[0], img.shape[1])
    if max_size > 1000:
        img = rescale(img, 1000. / max_size)
        img = (img * 255).astype(np.uint8)

    out_dir = '/home/khanhhh/data_1/projects/Oh/data/face/face_skin_color'
    imgs = extractor.extract(img,  sample_skin, hsv=False)

    for i, color_img in enumerate(imgs):
        cv.imwrite(filename=f'{out_dir}/cluster_color_{i}.jpg', img=color_img)