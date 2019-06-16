# import the necessary packages
import face_utils
import numpy as np
import argparse
import dlib
import cv2 as cv
import matplotlib.pyplot as plt
from skimage.transform import PiecewiseAffineTransform, warp
from common.obj_util import import_mesh_tex_obj
import pickle
import os
from collections import defaultdict

def translate(image, x, y):
    # define the translation matrix and perform the translation
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv.warpAffine(image, M, (image.shape[1], image.shape[0]))

    # return the translated image
    return shifted

def rotate(image, angle, center=None, scale=1.0):
    # grab the dimensions of the image
    (h, w) = image.shape[:2]

    # if the center is None, initialize it as the center of
    # the image
    if center is None:
        center = (w // 2, h // 2)

    # perform the rotation
    M = cv.getRotationMatrix2D(center, angle, scale)
    rotated = cv.warpAffine(image, M, (w, h))

    # return the rotated image
    return rotated


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv.warpAffine(image, M, (nW, nH))

def resize(image, width=None, height=None, inter=cv.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized

def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

def test_thin_plate():
    sources = []
    sources.append([0,0])
    sources.append([799,0])
    sources.append([0,599])
    sources.append([799,599])

    targets = []
    targets.append([100,0])
    targets.append([799,0])
    targets.append([0,599])
    targets.append([799,599])

    matches = list()
    for i in range(len(sources)):
        matches.append(cv.DMatch(i, i, 0))

    sources = np.array(sources).reshape((1, -1, 2)).astype(np.float32)
    targets = np.array(targets).reshape((1, -1, 2)).astype(np.float32)
    #sources = np.array(sources).T
    #targets = np.array(targets).T

    tps = cv.createThinPlateSplineShapeTransformer()
    tps.estimateTransformation(sources, targets, matches)
    retval, test_pnts = tps.applyTransformation(sources)
    print(test_pnts)
    print(np.mean(test_pnts - targets))

    tps = cv.createThinPlateSplineShapeTransformer()

    sshape = np.array([[67, 90], [206, 90], [67, 228], [206, 227]], np.float32)
    tshape = np.array([[64, 63], [205, 122], [67, 263], [207, 192]], np.float32)
    sshape = sshape.reshape(1, -1, 2)
    tshape = tshape.reshape(1, -1, 2)
    matches = list()
    matches.append(cv.DMatch(0, 0, 0))
    matches.append(cv.DMatch(1, 1, 0))
    matches.append(cv.DMatch(2, 2, 0))
    matches.append(cv.DMatch(3, 3, 0))
    tps.estimateTransformation(tshape, sshape, matches)
    ret, tshape = tps.applyTransformation(sshape)
    print(tshape)

def test():
    img_path = '/home/khanhhh/data_1/projects/Oh/data/face/prn_output/front_IMG_1933.jpg'
    image = cv.imread(img_path)
    sources = detect_face_landmark(image)
    sources = sources.astype(np.float32)
    sources  = sources.reshape((1,-1, 2))

    mkhm_img_tpl_path = "/home/khanhhh/data_1/projects/Oh/data/face/Make_Human_3D face/Head_FemaleGen01_Planar_01.png"
    img_mkhm = cv.imread(mkhm_img_tpl_path)
    targets = detect_face_landmark(img_mkhm)
    targets = targets.astype(np.float32)
    targets  = targets.reshape((1,-1, 2))

    matches = list()
    for i in range(targets.shape[1]):
        v = sources[0,i,:] - targets[0,i,:]
        d = np.linalg.norm(v)
        matches.append(cv.DMatch(i, i, d))

    tps = cv.createThinPlateSplineShapeTransformer()
    tps.estimateTransformation(sources, targets, matches)

    ret, test = tps.applyTransformation(sources)
    print(test-targets)

    test_img = image.copy()
    for i in range(targets.shape[1]):
        x,y = int(sources[0,i,0]), int(sources[0,i,1])
        cv.circle(test_img, (x, y), 1, (0, 0, 255), -1)
        cv.putText(test_img, f'{i}', (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 0), 1)

        x,y = int(targets[0,i,0]), int(targets[0,i,1])
        cv.circle(test_img, (x, y), 1, (255, 0, 0), -1)
        cv.putText(test_img, f'{i}', (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 0), 1)

    img_warp_mkhm = image.copy()
    tps.warpImage(image, img_warp_mkhm)
    #img_warp_mkhm = img_warp_mkhm[:512, :512, :]

    plt.imshow(img_warp_mkhm[:,:,::-1])
    plt.show()


def find_head_tex_verts(mesh, head_vert_idxs):
    head_vert_set = set(head_vert_idxs)
    head_tex_verts_set = set()
    ft = mesh['ft']
    faces = mesh['f']
    for f_idx, f in enumerate(faces):
        is_head = True
        for v_idx in f:
            if v_idx not in head_vert_set:
                is_head = False
                break
        if is_head:
            head_ft = ft[f_idx]
            for vt_idx in head_ft:
                head_tex_verts_set.add(vt_idx )

    return np.array([v for v in head_tex_verts_set])

def build_v_vt_map(faces, faces_tex):
    v_vt_map = defaultdict(set)
    for f, ft in zip(faces, faces_tex):
        for v, vt in zip(f, ft):
            v_vt_map[v].add(vt)

    result = []
    n = len(v_vt_map.keys())
    for i in range(n):
        result.append([vt for vt in v_vt_map[i]])

    return result


class HmFaceWarp():

    def __init__(self, meta_dir):
        self.N_Facial_LD = 68

        self.targets = self._load_target_texture_landmarks(meta_dir)
        self.targets *= 1024
        #TODO: forget why we have to invser the y coordinate
        self.targets[:, 1] = 1024 - self.targets[:,1]
        assert self.targets.shape[0] == self.N_Facial_LD

        self.targets = self.targets.reshape((1, -1, 2))

        self.detector = dlib.get_frontal_face_detector()
        shape_predictor_path = '/home/khanhhh/data_1/projects/Oh/codes/human_estimation/data/shape_predictor_68_face_landmarks.dat'
        self.predictor = dlib.shape_predictor(shape_predictor_path)

        self.matches = list()
        for i in range(self.N_Facial_LD):
            self.matches.append(cv.DMatch(i, i, 0))

        #debug
        # from pathlib import Path
        # import os
        # dir = '/media/khanhhh/42855ff5-574e-4a41-ad10-0f08087b0ff6/data_1/projects/Oh/data/face/2019-06-04-face/'
        # out_dir = '/media/khanhhh/42855ff5-574e-4a41-ad10-0f08087b0ff6/data_1/projects/Oh/data/face/2019-06-04-face-dlib-landmark/'
        # os.makedirs(out_dir, exist_ok=True)
        # for img_path in Path(dir).glob('*.*'):
        #     img = cv.imread(str(img_path))
        #     print(img_path)
        #     sources = self._detect_face_landmark(img)
        #     if sources is not None:
        #         for i in range(sources.shape[1]):
        #             x, y = int(sources[0, i, 0]), int(sources[0, i, 1])
        #             cv.circle(img, (x, y), 2, (0, 0, 255), -1)
        #     debug_path = f'{out_dir}/{img_path.stem}.jpg'
        #     cv.imwrite(debug_path, img)
        # exit()

    def warp(self, img):
        if isinstance(img, str):
            image = cv.imread(img)
        else:
            image = img

        #TODO: resize the image this way will ignore the normal aspect rate, which could affect the perfomance of dlib landmark detector
        #however, we need the original image to have the same size as the texture to be able to warp it.
        #is there anyway around?
        image = cv.resize(image, dsize=(1024, 1024), interpolation=cv.INTER_AREA)

        sources = self._detect_face_landmark(image)
        if sources is None:
            return image

        tps = cv.createThinPlateSplineShapeTransformer(0)
        tps.estimateTransformation(self.targets, sources, self.matches)

        tps.warpImage(image, image, flags=cv.INTER_CUBIC)

        #TODO: remove in release
        test = self.targets.copy().astype(np.int)
        image[test[0,:,1], test[0,:,0]] = (255, 0, 0)

        #TODO: remove. we make black pixels white for the sake of visualization
        #image[image[:,:,0] == 0] = (255,255,255)

        return image

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
            shape = face_utils.shape_to_np(shape)

            shape = shape.astype(np.float32)
            shape = shape.reshape((1, -1, 2))
            return shape

        return None

    def _load_target_texture_landmarks(self, meta_dir):
        mpath = os.path.join(*[meta_dir, 'victoria_template_textured_warped.obj'])
        mesh = import_mesh_tex_obj(mpath)
        verts_tex = mesh['vt']
        faces = mesh['f']
        faces_tex = mesh['ft']

        ld_idxs_path = os.path.join(*[meta_dir, 'victoria_face_landmarks.pkl'])
        with open(ld_idxs_path, 'rb') as file:
            vic_facial_ld_idxs_dict = pickle.load(file)
            vic_facial_ld_v_idxs = []
            for i in range(self.N_Facial_LD):
                vic_facial_ld_v_idxs.append(vic_facial_ld_idxs_dict[i])
            assert len(set(vic_facial_ld_v_idxs)) == len(vic_facial_ld_v_idxs)

        v_vt_map = build_v_vt_map(faces, faces_tex)
        vt_face_ld = []
        for i in range(self.N_Facial_LD):
            uv_tex_idxs = v_vt_map[vic_facial_ld_v_idxs[i]]
            assert len(uv_tex_idxs) == 1
            vt_face_ld.append(uv_tex_idxs[0])

        vt_face_ld_co = verts_tex[vt_face_ld]

        return vt_face_ld_co


class HmFPrnNetFaceTextureEmbedder():

    def __init__(self, prn_facelib_rect_path, texture_size = 1024):
        self.texture_size = texture_size
        self.rect_uv = np.loadtxt(prn_facelib_rect_path)
        self.rect = (texture_size * self.rect_uv).astype(np.int)
        self.rect[:,0] = self.texture_size - self.rect[:,0] - 1
        self.embed_size = self.rect.max() - self.rect.min()

        print(f'p0 = {self.rect[0,:]}')
        print(f'p1 = {self.rect[1,:]}')
        plt.plot(self.rect[0,0], self.rect[0,1], 'g+')
        plt.plot(self.rect[1,0], self.rect[1,1], 'r+')
        plt.show()

    def embed(self, prn_facelib_tex):

        assert prn_facelib_tex.shape[0] == prn_facelib_tex.shape[1], 'require square texture shape'

        texture = np.zeros(shape=(self.texture_size, self.texture_size, 3), dtype=np.uint8)

        if self.embed_size != prn_facelib_tex.shape[0]:
            prn_facelib_tex = cv.resize(prn_facelib_tex, (self.embed_size, self.embed_size), interpolation=cv.INTER_CUBIC)

        texture[self.rect[1,0]:self.rect[0,0], self.rect[0,1]:self.rect[1,1], :] = prn_facelib_tex

        # import  matplotlib.pyplot as plt
        # plt.imshow(texture)
        # plt.show()
        return texture

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i",  type=str, required=True)
    ap.add_argument("-o",  type=str, required=True)

    args = ap.parse_args()
    img_path = args.i
    out_path = args.o

    dir = '/home/khanhhh/data_1/projects/Oh/codes/human_estimation/data/meta_data/'
    face_warp = HmFaceWarp(dir)

    image = cv.imread(img_path)
    img_warp = face_warp.warp(image)

    cv.imwrite(filename=out_path, img=img_warp)

    # show the output image with the face detections + facial landmarks
    plt.subplot(121)
    plt.imshow(image[:,:, ::-1])
    plt.subplot(122)
    plt.imshow(img_warp[:,:,::-1])
    plt.show()

