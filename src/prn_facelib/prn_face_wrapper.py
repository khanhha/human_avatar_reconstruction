import numpy as np
import os
from glob import glob
import scipy.io as sio
from skimage.io import imread, imsave
from skimage.transform import rescale, resize
from time import time
import argparse
import ast

import cv2 as cv

from prn_facelib.api import PRN
from prn_facelib.utils.render_app import get_visibility, get_uv_mask, get_depth_image

class PrnFaceWrapper:
    def __init__(self, texture_size, prn_datadir_prefix):
        self.prn = PRN(is_dlib=True, prefix=prn_datadir_prefix)
        self.texture_size = texture_size
        pass

    def predict(self, image):
        """
        :param image:  RGB uint8 image
        """
        assert len(image.shape) ==3 and image.shape[2] == 3, 'incorrect image shape'

        [h, w, c] = image.shape

        max_size = max(image.shape[0], image.shape[1])
        if max_size > 1000:
            image = rescale(image, 1000. / max_size)
            image = (image * 255).astype(np.uint8)

        pos = self.prn.process(image)  # use dlib to detect face

        vertices = self.prn.get_vertices(pos)
        save_vertices = vertices.copy()
        save_vertices[:, 1] = h - 1 - save_vertices[:, 1]

        if self.texture_size != 256:
            pos_interpolated = resize(pos, (self.texture_size, self.texture_size), preserve_range=True)
        else:
            pos_interpolated = pos.copy()

        # test = pos_interpolated.round().astype(np.int32)
        # image_ldms = image.copy()
        # image_ldms[test[:, :,1], test[:,:,0]] = (0.0, 0, 0)
        # image_ldms[test[self.prn.uv_kpt_ind[1, :],self.prn.uv_kpt_ind[0, :],1], test[self.prn.uv_kpt_ind[1, :],self.prn.uv_kpt_ind[0, :],0]] = (255.0, 0, 0)
        #
        # image_ldms[test[145:155, 45:55, 1], test[145:155, 45:55, 0]] = (255, 255, 0)
        #
        # image[test[self.prn.uv_kpt_ind[1, :],self.prn.uv_kpt_ind[0, :],1], test[self.prn.uv_kpt_ind[1, :],self.prn.uv_kpt_ind[0, :],0]] = (255.0, 0, 0)

        img_face_points = pos_interpolated
        img_kpt = np.vstack([img_face_points[self.prn.uv_kpt_ind[1, :], self.prn.uv_kpt_ind[0, :], 0],
                             img_face_points[self.prn.uv_kpt_ind[1, :], self.prn.uv_kpt_ind[0, :], 1]]).T

        # set all pixels outside the convex hull of facial landmarks as black
        # convex_idxs = [i for i in range(17)] + [26,25,24] + [19,18,17]
        # polygon_pnts = img_kpt[convex_idxs, :].reshape(1,-1,2)
        # polygon_pnts = polygon_pnts.astype(np.int32)
        # mask = np.zeros(image.shape[:2], np.uint8)
        # cv.fillConvexPoly(mask, polygon_pnts, 1)
        # mask = mask.astype(np.bool)
        # mask = np.bitwise_not(mask)
        # image[mask,:] = (0,0,0)
        #
        # texture = cv.remap(image, pos_interpolated[:, :, :2].astype(np.float32), None, interpolation=cv.INTER_CUBIC,
        #                     borderMode=cv.BORDER_CONSTANT, borderValue=(0))
        # #debug
        # texture[self.prn.uv_kpt_ind[1, :], self.prn.uv_kpt_ind[0, :], :] = (255, 0, 0)

        return save_vertices, pos_interpolated[:,:,:2].astype(np.float32), img_kpt, image