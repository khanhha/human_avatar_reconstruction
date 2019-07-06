# import the necessary packages
import face_utils
import numpy as np
import argparse
import dlib
import cv2 as cv
import matplotlib.pyplot as plt
from skimage.transform import PiecewiseAffineTransform, warp
from common.obj_util import import_mesh_tex_obj, import_mesh_obj
from skimage import draw
import pickle
import os
from collections import defaultdict, Counter
from sklearn import mixture
from sklearn.cluster import KMeans
from src.deploy.multi_resol_texture_syn import  multiResolution_textureSynthesis
from src.deploy.patchBasedTextureSynthesis import patchBasedTextureSynthesis
from tex_syn.preprocess import *
from tex_syn.generate import *
import time
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

    def warp(self, img, img_face_kpt = None):
        if isinstance(img, str):
            image = cv.imread(img)
        else:
            image = img

        #TODO: currently we assume that the face complete like inside the rectangle of [0:1024, 0:1024]. need to handle it more accurately
        texture = np.zeros((1024, 1024, 3), dtype=np.uint8)
        w_max = min(texture.shape[1], img.shape[1])
        h_max = min(texture.shape[0], img.shape[0])
        texture[0:h_max, 0:w_max, :] = img[0:h_max, 0:w_max,:]

        if img_face_kpt is None:
            sources = self._detect_face_landmark(texture)
            if sources is None:
                return image
        else:
            sources = img_face_kpt.reshape((1, -1, 2))

        tps = cv.createThinPlateSplineShapeTransformer(0)
        tps.estimateTransformation(self.targets, sources, self.matches)

        tps.warpImage(texture, texture, flags=cv.INTER_CUBIC)

        #TODO: remove in release
        test = self.targets.copy().astype(np.int)
        texture[test[0,:,1], test[0,:,0]] = (0, 0, 255)

        #TODO: remove in release
        # test_1 = sources.copy().astype(np.int)
        # texture[test_1[0,:,1], test_1[0,:,0]] = (255, 0, 0)

        #TODO: remove. we make black pixels white for the sake of visualization
        #texture[texture[:,:,0] == 0] = (255,255,255)

        # plt.imshow(texture)
        # plt.show()

        return texture

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

    def __init__(self, meta_dir, prn_facelib_rect_path, texture_size = 1024):
        self.texture_size = texture_size
        self.rect_uv = np.loadtxt(prn_facelib_rect_path)
        #TODO: we assume the embed texture area is a square here
        self.embed_size  = np.round(0.5*texture_size * self.rect_uv[1,:]).astype(np.int)[0]
        self.rect_center = self.rect_uv[0,:]
        self.rect_center[1] = 1.0 - self.rect_center[1]
        self.rect_center = np.round(texture_size * self.rect_center).astype(np.int)

        vic_tpl_vertex_groups_path = os.path.join(*[meta_dir, 'victoria_part_vert_idxs.pkl'])

        vic_tpl_face_mesh_path = os.path.join(*[meta_dir, 'vic_mesh_textured_warped.obj'])
        vic_mesh = import_mesh_tex_obj(vic_tpl_face_mesh_path)
        verts_tex = vic_mesh['vt']
        faces_tex = vic_mesh['ft']
        vic_tri_quads = vic_mesh['f']

        with open(vic_tpl_vertex_groups_path, 'rb') as file:
            vparts = pickle.load(file=file)
            vface_map = vparts['face']

        non_face_tris_idxs = self._faces_inverse_from_verts(vic_tri_quads, vface_map)

        self.non_face_tex_mask = np.zeros((texture_size, texture_size), dtype=np.bool)
        for idx in non_face_tris_idxs:
            ft = faces_tex[idx]
            ft_uvs = []
            for v_ft in ft:
                uv = verts_tex[v_ft, :]
                ft_uvs.append(uv)
            ft_uvs = np.array(ft_uvs)
            ft_uvs[:,1] = 1.0 - ft_uvs[:,1]
            ft_uvs = np.round((self.texture_size * ft_uvs))
            rr, cc = draw.polygon(ft_uvs[:,1], ft_uvs[:,0])
            self.non_face_tex_mask[rr, cc] = 0.5

        face_tris_idxs = self._faces_from_verts(vic_tri_quads, vface_map)
        self.face_tex_mask = np.zeros((texture_size, texture_size), dtype=np.bool)
        for idx in face_tris_idxs:
            ft = faces_tex[idx]
            ft_uvs = []
            for v_ft in ft:
                uv = verts_tex[v_ft, :]
                ft_uvs.append(uv)
            ft_uvs = np.array(ft_uvs)
            ft_uvs[:,1] = 1.0 - ft_uvs[:,1]
            ft_uvs = np.round((self.texture_size * ft_uvs))
            rr, cc = draw.polygon(ft_uvs[:,1], ft_uvs[:,0])
            self.face_tex_mask[rr, cc] = 0.5

        # self.non_face_tex_mask[self.rect_center[1]-10:self.rect_center[1]+10, self.rect_center[0]-10:self.rect_center[0]+10] = 0.2
        # plt.imshow(self.face_tex_mask)
        # plt.show()


        self.N_Facial_LD = 68

        self.targets = self._load_target_texture_landmarks(meta_dir)
        self.targets *= 1024
        #TODO: forget why we have to invser the y coordinate
        self.targets[:, 1] = 1024 - self.targets[:,1]
        assert self.targets.shape[0] == self.N_Facial_LD

        self.targets = self.targets.reshape((1, -1, 2))

        self.matches = list()
        for i in range(self.N_Facial_LD):
            self.matches.append(cv.DMatch(i, i, 0))

    def _faces_from_verts(self, all_faces, verts):
        """
        find all the faces each of which does contain all vertices in verts
        :param all_faces:
        :param verts:
        :return:
        """
        vset = set(verts)
        faces = []
        for f_idx, f in enumerate(all_faces):
            in_group = True
            for v in f:
                if v not in vset:
                    in_group = False
                    break

            if in_group == True:
                faces.append(f_idx)

        return faces

    def _faces_inverse_from_verts(self, all_faces, verts):
        """
        find all the faces each of which does not contain all vertices in verts
        :param all_faces:
        :param verts:
        :return:
        """
        vset = set(verts)
        faces = []
        for f_idx, f in enumerate(all_faces):
            in_group = True
            for v in f:
                if v not in vset:
                    in_group = False
                    break

            if in_group == False:
                faces.append(f_idx)

        return faces

    def synthesize_skin_color_1(self, texture):
        img_path = '/home/khanhhh/data_1/projects/Oh/data/face/2019-06-04-face/MVIMG_20190604_180645.jpg'
        skin_sample = cv.imread(img_path)
        skin_sample = skin_sample[2500:2900,1300:1800, :]
        return cv.resize(skin_sample, dsize=(self.texture_size, self.texture_size))

    def synthesize_skin_color_tiling(self, face_texture):
        #path = '/home/khanhhh/data_1/projects/Oh/data/face/skin_texture_patch.jpg'
        #path = '/home/khanhhh/data_1/projects/Oh/data/face/skin_texture_patch_cheek.jpg'
        #path = '/home/khanhhh/data_1/projects/Oh/data/face/skin_texture_patch_forehead_small.jpg'
        path = '/home/khanhhh/data_1/projects/Oh/data/face/skin_texture_patch_forehead.jpg'
        exampleMapPath = path
        outputPath = "out/1/"
        patchSize =  16  # size of the patch (without the overlap)
        overlapSize = 4# the width of the overlap region
        outputSize = [1024, 1024]
        patch = face_texture[1:35,90:160, :][:,:,::-1]
        patch = patch.astype(np.float) / 255.0
        pbts = patchBasedTextureSynthesis(patch, outputPath, outputSize, patchSize, overlapSize,
                                          in_windowStep=5, in_mirror_hor=False, in_mirror_vert=False, in_shapshots=False)
        img_texture = pbts.resolveAll()
        img_texture = img_texture[:,:,::-1]

        return img_texture

    def synthesize_skin_color_quilting(self, face_texture):
        patch = face_texture[1:35,90:160, :][:,:,::-1]
        patch = patch / 255.0

        block_size = 20
        overlap = int(block_size/6.0)
        tolerance = 0.1
        outH = self.texture_size
        outW = self.texture_size
        start_time = time.time()
        img_texture = generateTextureMap(patch, block_size, overlap, outH, outW, tolerance)
        end_time = time.time()
        print('quilting time: ', end_time-start_time)

        img_texture = img_texture[:,:,::-1]

        print(img_texture.shape, img_texture.dtype)
        img_texture = (img_texture * 255.0).astype(np.uint8)
        img_texture = cv.resize(img_texture, (self.texture_size, self.texture_size), interpolation=cv.INTER_AREA)
        plt.imshow(img_texture[:,:,::-1])
        plt.show()
        return img_texture

    def synthesise_skin_color_seamlesscloneing(self, face_texture):
        patch = face_texture[1:35,90:160, :]
        skin_texture = np.zeros((self.texture_size, self.texture_size, 3), np.uint8)
        for i in range(0, self.texture_size, patch.shape[0]):
            for j in range(0, self.texture_size, patch.shape[1]):
                i1 = min(i+patch.shape[0], self.texture_size)
                j1 = min(j+patch.shape[1], self.texture_size)
                skin_texture[i:i1, j:j1, :] = patch[0:i1-i, 0:j1-j, :]

        skin_texture_1 = skin_texture.copy()
        mask = np.ones((patch.shape[0], patch.shape[1]), dtype=np.uint8)*255
        mask[:4,:] = 0
        mask[:,:4] = 0
        mask[:,-4:] = 0
        mask[-4:,:] = 0
        # plt.imshow(patch[:, :, ::-1])
        # plt.imshow(mask, alpha=0.5)
        # plt.show()

        for i in range(0, self.texture_size, patch.shape[0]):
            for j in range(0, self.texture_size, patch.shape[1]):
                i1 = min(i+patch.shape[0], self.texture_size)
                j1 = min(j+patch.shape[1], self.texture_size)
                x = int((j+j1)/2)
                y = int((i+i1)/2)
                skin_texture_1 = cv.seamlessClone(patch, skin_texture_1, mask, (x,y), cv.MIXED_CLONE)

                cv.drawMarker(skin_texture_1, (x,y), (255,0,0), markerType=cv.MARKER_SQUARE, markerSize=5, thickness=5)
                plt.subplot(121)
                plt.imshow(skin_texture[:,:,::-1])
                plt.subplot(122)
                plt.imshow(skin_texture_1[:,:,::-1])
                plt.show()
        return skin_texture

    def synthesize_skin_color_kmeans(self, texture):
        img_path = '/home/khanhhh/data_1/projects/Oh/data/face/2019-06-04-face/MVIMG_20190604_180645.jpg'
        skin_sample = cv.imread(img_path)
        #skin_sample = skin_sample[1800:2000,1620:1820, :]
        skin_sample = skin_sample[2500:2900,1300:1800, :]

        # Initiate KMeans Object
        estimator = KMeans(n_clusters=5, random_state=0)
        estimator.fit(np.reshape(skin_sample, (-1,3)))

        occurance_counter = Counter(estimator.labels_)
        # Get the total sum of all the predicted occurances
        totalOccurance = sum(occurance_counter.values())

        colorInformation = []
        clusters = estimator.cluster_centers_
        # Loop through all the predicted colors
        for x in occurance_counter.most_common(len(clusters)):
            index = (int(x[0]))

            # Get the color number into a list
            color = clusters[index].tolist()

            # Get the percentage of each color
            color_percentage = (x[1] / totalOccurance)

            # make the dictionay of the information
            colorInfo = {"cluster_index": index, "color": color, "color_percentage": color_percentage}

            # Add the dictionary to the list
            colorInformation.append(colorInfo)

        print(colorInformation)
        sk_color = colorInformation[0]['color']
        sk_color = np.array(sk_color).astype(np.uint8)
        syn_tex = np.zeros((self.texture_size, self.texture_size, 3), np.uint8)
        syn_tex[:] = sk_color

        plt.subplot(131)
        plt.imshow(skin_sample[:,:,::-1])
        plt.subplot(132)
        plt.imshow(syn_tex[:,:,::-1])
        plt.subplot(133)
        plt.imshow(texture[:,:,::-1])
        plt.show()

        return syn_tex

    def synthesize_skin_texture_base_color(self, face_texture):
        #titling from the original image
        #img_path = '/home/khanhhh/data_1/projects/Oh/data/face/2019-06-04-face/MVIMG_20190604_180645.jpg'
        #skin_sample = cv.imread(img_path)
        #skin_sample = skin_sample[1800:2000,1620:1820, :]  \
        #img_path = '/home/khanhhh/data_1/projects/Oh/data/face/skin_texture_patch_forehead.jpg'
        #G0 = cv.imread(img_path)
        G0 = face_texture[1:35,90:160, :]
        gpB = [G0]
        G = G0.copy()
        for i in range(10):
            G = cv.pyrDown(G)
            gpB.append(G)
            if G.shape[0] == 1:
                break

        G = gpB[-1]
        skin_texture = np.zeros((self.texture_size, self.texture_size, 3), dtype=np.uint8)
        skin_texture[:,:,:] = G[0,0,:]
        # plt.subplot(121)
        # plt.imshow(G0[:, :, ::-1])
        # plt.subplot(122)
        # plt.imshow(skin_texture[:,:,::-1])
        # plt.show()

        return skin_texture
        # img_path = '/home/khanhhh/data_1/projects/Oh/data/face/skin_texture_patch_forehead_base_color.jpg'
        # skin_patch = cv.imread(img_path)
        # tiling = False
        # if tiling:
        #     skin_texture = np.zeros((self.texture_size, self.texture_size, 3), np.uint8)
        #     for i in range(0, self.texture_size, skin_patch.shape[0]):
        #         for j in range(0, self.texture_size, skin_patch.shape[1]):
        #             i1 = min(i+skin_patch.shape[0], self.texture_size)
        #             j1 = min(j+skin_patch.shape[1], self.texture_size)
        #             skin_texture[i:i1, j:j1, :] = skin_patch[0:i1-i, 0:j1-j, :]
        # else:
        #     skin_texture = cv.resize(skin_patch, (self.texture_size, self.texture_size), interpolation=cv.INTER_AREA)
        #
        # return skin_texture

    def _load_target_texture_landmarks(self, meta_dir):
        mpath = os.path.join(*[meta_dir, 'vic_mesh_textured_warped.obj'])
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

    def warp(self, image_rgb_front, image_face_landmarks):
        #TODO: currently we assume that the face complete like inside the rectangle of [0:1024, 0:1024]. need to handle it more accurately
        texture = np.zeros((1024, 1024, 3), dtype=np.uint8)
        w_max = min(texture.shape[1], image_rgb_front.shape[1])
        h_max = min(texture.shape[0], image_rgb_front.shape[0])
        texture[0:h_max, 0:w_max, :] = image_rgb_front[0:h_max, 0:w_max,:]

        sources = image_face_landmarks.reshape((1, -1, 2))

        tps = cv.createThinPlateSplineShapeTransformer(10)
        tps.estimateTransformation(self.targets, sources, self.matches)

        tps.warpImage(texture, texture, flags=cv.INTER_CUBIC)

        #TODO: remove in release
        test = self.targets.copy().astype(np.int)
        texture[test[0,:,1], test[0,:,0]] = (0, 0, 255)

        #TODO: remove in release
        # test_1 = sources.copy().astype(np.int)
        # texture[test_1[0,:,1], test_1[0,:,0]] = (255, 0, 0)

        #TODO: remove. we make black pixels white for the sake of visualization
        #texture[texture[:,:,0] == 0] = (255,255,255)

        # plt.imshow(texture)
        # plt.show()

        return texture

    def embed(self, prn_facelib_tex):

        assert prn_facelib_tex.shape[0] == prn_facelib_tex.shape[1], 'require square texture shape'

        face_texture = np.zeros(shape=(self.texture_size, self.texture_size, 3), dtype=np.uint8)

        if (2*self.embed_size + 1) != prn_facelib_tex.shape[0]:
            prn_facelib_tex = cv.resize(prn_facelib_tex, (2*self.embed_size + 1, 2*self.embed_size + 1), interpolation=cv.INTER_CUBIC)

        # #debug
        # prn_facelib_tex[:2, :, :] = (255,0,0)
        # prn_facelib_tex[:, :2, :] = (255,0,0)
        # prn_facelib_tex[-2:,:, :] = (255,0,0)
        # prn_facelib_tex[:,-2:, :] = (255,0,0)

        face_texture[self.rect_center[1]-self.embed_size:self.rect_center[1]+self.embed_size+1, self.rect_center[0]-self.embed_size:self.rect_center[0]+self.embed_size+1, :] = prn_facelib_tex

        # skin_sample = np.reshape(skin_sample, newshape=(-1,3))
        # rand_idxs = np.random.randint(0, skin_sample.shape[0], self.texture_size*self.texture_size)
        # skin_sample_1 = skin_sample[rand_idxs, :]
        # skin_sample_1.shape=(self.texture_size, self.texture_size, 3)


        # upsample approach
        # img_path = '/home/khanhhh/data_1/projects/Oh/data/face/2019-06-04-face/MVIMG_20190604_180645.jpg'
        # skin_sample = cv.imread(img_path)
        # skin_sample = skin_sample[1800:2000,1620:1820, :]
        # skin_sample_1 = cv.resize(skin_sample, dsize=(self.texture_size, self.texture_size))

        #sample from texture
        #skin_sample = texture[430:490, 570:620, :] #cheek
        # skin_sample = texture[340:380, 510:560, :] #forehead
        # skin_sample_1 = np.zeros((self.texture_size, self.texture_size, 3), np.uint8)
        # for i in range(0, self.texture_size, skin_sample.shape[0]):
        #      for j in range(0, self.texture_size, skin_sample.shape[1]):
        #          i1 = min(i+skin_sample.shape[0], self.texture_size)
        #          j1 = min(j+skin_sample.shape[1], self.texture_size)
        #          skin_sample_1[i:i1, j:j1, :] = skin_sample[0:i1-i, 0:j1-j, :]

        #skin_sample_1 = self.synthesize_skin_color(texture)
        #skin_sample_1 = self.synthesize_skin_color_1(texture)

        #head_texture = self.synthesize_skin_color_2(face_texture)
        #cv.imwrite('/home/khanhhh/data_1/projects/Oh/data/face/test_merge_face_body/test_f_victoria_head_tex_synthesis.jpg', head_texture)

        #face_texture[np.bitwise_not(self.face_tex_mask), :] = head_texture[np.bitwise_not(self.face_tex_mask), :]
        #return face_texture

        mask = self.face_tex_mask.astype(np.uint8) * 255
        mask_rect = cv.boundingRect(mask)
        mask_center = (mask_rect[0]+int(mask_rect[2]/2), mask_rect[1]+int(mask_rect[3]/2))
        # print(mask_rect)
        # cv.drawMarker(face_texture, (mask_rect[0]+int(mask_rect[2]/2), mask_rect[1]+int(mask_rect[3]/2)), (0,0,255), markerSize=5, thickness=2)
        # plt.imshow(face_texture)
        # plt.imshow(mask, alpha=0.5)
        # plt.show()
        # exit()
        prn_mask = mask[self.rect_center[1]-self.embed_size:self.rect_center[1]+self.embed_size+1, self.rect_center[0]-self.embed_size:self.rect_center[0]+self.embed_size+1]
        head_texture = self.synthesize_skin_texture_base_color(prn_facelib_tex)
        #head_texture = self.synthesize_skin_color_tiling(prn_facelib_tex)
        #head_texture = self.synthesise_skin_color_seamlesscloneing(prn_facelib_tex)
        # tmp_path = '/home/khanhhh/data_1/projects/Oh/data/face/test_merge_face_body/tmp.png'
        # from pathlib import Path
        # if not Path(tmp_path).exists():
        #     head_texture = self.synthesize_skin_color_quilting(prn_facelib_tex)
        #     cv.imwrite(tmp_path, head_texture)
        # else:
        #     head_texture = cv.imread(tmp_path)

        # head_texture_color_base_1 = cv.cvtColor(head_texture_color_base, cv.COLOR_BGR2HSV)
        # head_texture_1 = cv.cvtColor(head_texture, cv.COLOR_BGR2HSV)
        # head_texture_color_base_1 = head_texture_color_base.copy()
        # head_texture_1 = head_texture.copy()
        #
        # head_texture_1 = 0.5 *( head_texture_color_base_1 + head_texture_1)
        # head_texture_1 = head_texture_1.astype(np.uint8)
        # head_texture_1 = cv.cvtColor(head_texture_1, cv.COLOR_HSV2BGR)
        #
        # plt.subplot(131)
        # plt.imshow(head_texture_color_base[:,:,::-1])
        # plt.subplot(132)
        # plt.imshow(head_texture[:,:,::-1])
        # plt.subplot(133)
        # plt.imshow(head_texture_1[:,:,::-1])
        # plt.show()
        #head_texture = head_texture_1

        start_time = time.time()
        head_texture = cv.seamlessClone(prn_facelib_tex, head_texture, prn_mask, mask_center, cv.NORMAL_CLONE)
        elapsed_time = time.time() - start_time
        print('seamless time: ', elapsed_time)

        #head_texture[self.rect_center[1]-self.embed_size:self.rect_center[1]+self.embed_size+1, self.rect_center[0]-self.embed_size:self.rect_center[0]+self.embed_size+1, :] = prn_facelib_tex

        return head_texture

        # height, width, channels = head_texture.shape
        # center = (int(width / 2), int(height / 2))
        # head_texture = cv.seamlessClone(face_texture, head_texture, mask, center, cv.NORMAL_CLONE)
        # cv.drawMarker(head_texture, center, (0,0,255), markerSize=10, thickness=5)
        # plt.imshow(face_texture[:,:,::-1])
        # plt.imshow(head_texture[:,:,::-1], alpha=0.5)
        # plt.show()
        #return head_texture

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

