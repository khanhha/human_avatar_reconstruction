# import the necessary packages
import face_utils
import numpy as np
import dlib
import cv2 as cv
import matplotlib.pyplot as plt
import os
from pathlib import Path
from face_parse.face_parser import FaceParser

# this class does 3 thing
# cut the facial region from the input image. resize it so that its width is equal to resize_face_width
# predict 68 facial landmarks
# predict face segmentation map that defines lips, nose, eye-brows regions.
class FaceExtractor():

    def __init__(self, model_dir, resize_face_width = 512):
        self.img_width = resize_face_width
        self.face_detector = dlib.get_frontal_face_detector()
        shape_predictor_path = os.path.join(*[model_dir, 'dlib_shape_predictor_68_face_landmarks.dat'])
        assert Path(shape_predictor_path).exists(), "dlib_shape_predictor_68_face_landmarks.dat does not exist at this path: " + shape_predictor_path
        self.face_landmarks_predictor = dlib.shape_predictor(shape_predictor_path)

        # load face segmentation model
        # we need the mask of face without eye, mouth for estimating skin color
        self.face_parser = FaceParser(model_dir=model_dir)

    def _detect_face_landmark(self, img):
        """
        :param img: RGB image
        :return: 68x2 landamrks
        """
        gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

        # detect faces in the grayscale image
        rects = self.face_detector(gray, 1)
        # loop over the face detections
        for (i, rect) in enumerate(rects):
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = self.face_landmarks_predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            shape = shape.astype(np.float32)
            shape = shape.reshape((-1, 2))
            return shape

        return None

    def extract(self, img, debug_name=None):
        """
        :param img: RGB image
        :param debug_name: for debugging
        :return: face image that contains just facial region, 68x2 facial landamrks, segmentation map
        """
        ldms = self._detect_face_landmark(img)
        if ldms is None:
            return None

        ldms  = ldms.astype(np.int32)

        if debug_name is not None:
            debug_img = img.copy()
            for i in range(68):
                cv.circle(debug_img, (ldms[i, 0], ldms[i, 1]), 1, (255, 0, 0), thickness=cv.FILLED)

        left  = ldms[0,:]  + (0.5*(ldms[0,:] - ldms[27,:])).astype(np.int32)
        right = ldms[16,:] + (0.5*(ldms[16,:] - ldms[27,:])).astype(np.int32)
        top   = ldms[27,:] + (ldms[27,:] - ldms[8,:])
        bottom   = ldms[8,:] + (ldms[8,:] - ldms[33,:])
        bound_pnts = np.vstack([left, right, top, bottom])
        miny, maxy = max(bound_pnts[:,1].min(),0), min(bound_pnts[:,1].max(), img.shape[0])
        minx, maxx = max(bound_pnts[:,0].min(),0), min(bound_pnts[:,0].max(), img.shape[1])
        img_1 = img[miny:maxy, minx:maxx, :]

        # scale the face image so that its width is equal to sefl.img_width
        # we don't to scale both dimensions to avoid breaking the ratio of the face
        old_shape = img_1.shape[:]
        scale_ratio = self.img_width/old_shape[1]
        img_1 = cv.resize(img_1, (0,0), fx=scale_ratio, fy=scale_ratio, interpolation=cv.INTER_AREA)

        #transform ldms to new image space
        ldms[:,0] = ((ldms[:,0]-minx)/old_shape[1]) * img_1.shape[1]
        ldms[:,1] = ((ldms[:,1]-miny)/old_shape[0]) * img_1.shape[0]

        face_seg = self.face_parser.parse_face(img_1)

        if debug_name is not None:
            debug_dir = '/home/khanhhh/data_1/projects/Oh/data/face/google_front_faces/debug_face_extract'
            for i in range(68):
                cv.circle(img_1, (ldms[i,0],ldms[i,1]), 1, (255,0,0), thickness=cv.FILLED)
            os.makedirs(debug_dir, exist_ok=True)
            plt.subplot(121)
            plt.imshow(debug_img[:,:,::-1])
            plt.subplot(122)
            plt.imshow(img_1[:,:,::-1])
            plt.imshow(face_seg, alpha=0.5)
            plt.savefig(f'{debug_dir}/{debug_name}.png', dpi=500)

        return img_1, ldms, face_seg

if __name__ == '__main__':
    dir = '/home/khanhhh/data_1/projects/Oh/data/face/google_front_faces/'
    model_dir = '/home/khanhhh/data_1/projects/Oh/data/3d_human/deploy_models_nosyn/'
    extractor = FaceExtractor(model_dir)
    for path in Path(dir).glob('*.jpg'):
        img = cv.imread(str(path))
        extractor.extract(img, path.stem)

