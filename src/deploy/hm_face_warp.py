# import the necessary packages
import face_utils
import dlib
import cv2 as cv
from common.obj_util import import_mesh_tex_obj
from skimage import draw
from collections import defaultdict
from sklearn.mixture import BayesianGaussianMixture
from tex_syn.generate import *
from deploy.data_config import config_get_data_path

G_debug_id = 0

#cloning, alpha, pyramid
#G_blending_alg = 'cloning'
#G_blending_alg = 'alpha'
G_blending_alg = 'pyramid'

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


# this class has following features
# estimate skin color
# replace hair, background in the input image with estimated skin color
# map the input image to the PRN texture using the PRN texture map
# embed the PRN texture map to Victoria texture, which has larger size and also contains body region
class HmFPrnNetFaceTextureEmbedder():

    def __init__(self, meta_dir, texture_size = 1024):
        self.texture_size = texture_size
        #prn_facelib_rect_path = os.path.join(*[meta_dir, 'prn_texture_in_victoria_texture.txt'])
        #assert Path(prn_facelib_rect_path).exists(), f"there is no file prn_texture_in_victoria_texture.txt in meta_dir {meta_dir}"
        prn_facelib_rect_path = config_get_data_path(meta_dir, "prn_texture_in_victoria_texture")
        self.rect_uv = np.loadtxt(prn_facelib_rect_path)

        # TODO: we assume the embed texture area is a square here
        # size of PRN texture in Victoria texture
        self.embed_size  = np.round(0.5*self.texture_size * self.rect_uv[1,:]).astype(np.int)[0]
        # center of PRN texture in Victoria texture
        self.rect_center = self.rect_uv[0,:]
        self.rect_center[1] = 1.0 - self.rect_center[1]
        self.rect_center = np.round(self.texture_size * self.rect_center).astype(np.int)

        #vic_tpl_vertex_groups_path = os.path.join(*[meta_dir, 'victoria_part_vert_idxs.pkl'])
        vic_tpl_vertex_groups_path = config_get_data_path(meta_dir, 'victoria_part_vert_idxs')
        #vic_tpl_face_mesh_path = os.path.join(*[meta_dir, 'vic_mesh_textured_warped.obj'])
        vic_tpl_face_mesh_path = config_get_data_path(meta_dir, 'victoria_template_textured_mesh')
        self._build_face_texture_masks(vic_tpl_vertex_groups_path, vic_tpl_face_mesh_path)

    def _build_face_texture_masks(self, vic_tpl_vertex_groups_path, vic_tpl_face_mesh_path):
        """
        build masks for different regions inside the Victoria texture
        :param vic_tpl_vertex_groups_path: path to pickle files that contains vertex indices of different parts.
        :param vic_tpl_face_mesh_path: path to textured victoria template mesh
        """
        vic_mesh = import_mesh_tex_obj(vic_tpl_face_mesh_path)
        verts_tex = vic_mesh['vt']
        faces_tex = vic_mesh['ft']
        vic_tri_quads = vic_mesh['f']

        with open(vic_tpl_vertex_groups_path, 'rb') as file:
            vparts = pickle.load(file=file)
            vface_map = vparts['face']

        non_face_tris_idxs = self._faces_inverse_from_verts(vic_tri_quads, vface_map)

        # draw face mask using face triangles
        face_tris_idxs = self._faces_from_verts(vic_tri_quads, vface_map)
        self.face_tex_mask = np.zeros((self.texture_size, self.texture_size), dtype=np.bool)
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
            self.face_tex_mask[rr, cc] = True

        # draw non-face mask using non-face triangles
        self.non_face_tex_mask = np.zeros((self.texture_size, self.texture_size), dtype=np.bool)
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
            self.non_face_tex_mask[rr, cc] = True

        # self.non_face_tex_mask[self.rect_center[1]-10:self.rect_center[1]+10, self.rect_center[0]-10:self.rect_center[0]+10] = 0.2
        # plt.imshow(self.face_tex_mask)
        # plt.show()

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

    @staticmethod
    def estimate_skin_color(face_img, face_img_seg, take_only_best = True, n_clusters = 5, hsv=False):
        """
        estimate skin colors using clustering method from pixel colors inside facial_mask
        :param face_img: input BGR face image
        :param face_img_seg: face segmentation, returned by the face parsing model
        :param hsv: apply clustering on hsv color space or not
        :return: color of largest cluster
        """
        if hsv:
            face_img = cv.cvtColor(face_img, cv.COLOR_BGR2Lab)

        facial_mask = (face_img_seg == 1)
        colors = face_img[facial_mask, :]

        cov_type = 'full'
        #gmm = GaussianMixture(n_components=n_classes, covariance_type=cov_type, max_iter=20, random_state=0)
        gmm = BayesianGaussianMixture(n_components=n_clusters, covariance_type=cov_type, max_iter=20, random_state=0)
        cls_preds = gmm.fit_predict(colors)
        sorted_idxs = np.argsort(-gmm.weights_)

        result_colors = []
        #for the sake of performance
        if take_only_best:
            best_idx = sorted_idxs[0]
            best_colors_mask = cls_preds == best_idx
            best_colors = colors[best_colors_mask]
            best_color = np.mean(best_colors, axis=0).astype(np.uint8)
            result_colors.append(best_color)
        else:
            for i in range(n_clusters):
                cls_mask = (cls_preds == sorted_idxs[i])
                cls_color = np.mean(colors[cls_mask], axis=0).astype(np.uint8)
                result_colors.append(cls_color)

        debug = False
        #plot estimated color as background in the face image
        if debug:
            dir = '/home/khanhhh/data_1/projects/Oh/data/face/google_front_faces/debug_skin_colors/'
            os.makedirs(dir, exist_ok=True)
            mask_nonface = np.bitwise_not(facial_mask)

            out_imgs = []
            for i in sorted_idxs:
                cls_color = gmm.means_[i, :].astype(np.uint8)
                img_1 = face_img.copy()
                img_1[mask_nonface] = cls_color
                out_imgs.append(img_1)
                # plt.subplot(121)
                # plt.imshow(face_img)
                # plt.imshow(facial_mask, alpha=0.2)
                # plt.subplot(122)
                # plt.imshow(img_1)
                # plt.show()
            if hsv:
                for i in range(len(out_imgs)):
                    img = out_imgs[i]
                    img = cv.cvtColor(img, cv.COLOR_Lab2BGR)
                    out_imgs[i] = img

            for i, img in enumerate(out_imgs):
                cv.imwrite(f'{dir}/{G_debug_id}_{i}.jpg', img[:,:,::-1])

        return result_colors

    def synthesize_skin_texture_base_color(self, face_texture):
        """
        apply gaussian pyramid to the lowest level on a rectangular patch to find the base skin color
        however, this function is no in use anymore because it is sensntive to the selection of rectangular patch
        we are currently use clustering on every possible skin pixels that we can detect
        :param face_texture:  PRN texture
        :return:
        """
        #titling from the original image
        # manually select a rectangular patch
        G0 = face_texture[80:125, 150:200,:] #cheek
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
        return skin_texture

    def _blend_images_opt(self, A, B, m, level=5, debug=False):
        """
        blend images: optimized version
        :param A: face image
        :param B: normally background image filled in with estiamted skin color
        :param m: masking that defie the facial region in the face image
        :param level: the level of Laplacin pyramid for blending
        :param debug:
        :return:
        """
        A = A.astype(np.float32) / 255.0
        B = B.astype(np.float32) / 255.0

        N = level
        # generate Gaussian pyramid for A,B and mask
        GA = A.copy()
        GB = B.copy()
        GM = m.copy()
        LS = []
        for i in range(N):
            GA_down = cv2.pyrDown(GA)
            GB_down = cv2.pyrDown(GB)
            GM_down  = cv2.pyrDown(GM)

            GA_up = cv2.pyrUp(GA_down, dstsize=(GA.shape[1], GA.shape[0]))
            GB_up = cv2.pyrUp(GB_down, dstsize=(GB.shape[1], GB.shape[0]))
            LA = np.subtract(GA, GA_up)
            LB = np.subtract(GB, GB_up)
            bld_weights = np.dstack([GM, GM, GM])
            ls = LA * bld_weights + LB * (1.0 - bld_weights)
            LS.append(ls)

            GM = GM_down
            GA = GA_down
            GB = GB_down

        # add the bottom gaussian layer
        bld_weights = np.dstack([GM, GM, GM])
        ls = GA * bld_weights + GB * (1.0 - bld_weights)
        LS.append(ls)

        # now reconstruct
        LS.reverse()
        ls_ = LS[0]
        for i in range(1, N+1):
            ls_ = cv2.pyrUp(ls_, dstsize=(LS[i].shape[1], LS[i].shape[0]))

            if debug:
                plt.subplot(131)
                plt.imshow(LS[i])
                plt.subplot(132)
                plt.imshow(ls_)

            ls_ = ls_ + LS[i]

            if debug:
                plt.subplot(133)
                plt.imshow(ls_)
                plt.show()

        ls_ = np.clip(ls_ * 255.0, 0, 255.0).astype(np.uint8)

        if debug:
            plt.subplot(131)
            plt.imshow(A)
            plt.subplot(132)
            plt.imshow(A)
            plt.imshow(m, alpha=0.3)
            plt.subplot(133)
            plt.imshow(ls_)
            plt.show()

        return ls_

    def _fix_hair_and_background(self, face_img, face_seg, skin_color, blend_mode ='pyramid'):
        """
        :param face_img: face image
        :param face_seg: face segmentation map from the Pytorch face parsing model
        :param blend_mode: for testing mode. should be only 'pyramid'
        :return:
        """
        #1:face
        #2,3,4,5: eyes
        #10,11,12: nose, two lips
        final_mask = np.bitwise_or(np.bitwise_and(face_seg >= 1,face_seg <= 5), np.bitwise_and(face_seg >= 10,face_seg <= 13))
        final_mask = final_mask.astype(np.uint8) * 255
        face_bgr = np.empty_like(face_img)
        face_bgr[:,:,:] = skin_color

        if blend_mode == 'cloning':
            face_mask_rect   = cv.boundingRect(final_mask)
            face_mask_center = (face_mask_rect[0]+int(face_mask_rect[2]/2), face_mask_rect[1]+int(face_mask_rect[3]/2))
            face_img = cv.cvtColor(face_img, cv.COLOR_RGB2HSV_FULL)
            face_bgr = cv.cvtColor(face_bgr, cv.COLOR_RGB2HSV_FULL)
            face_img_1 = cv.seamlessClone(face_img, face_bgr, final_mask, face_mask_center, cv.NORMAL_CLONE)
            face_img_1 = cv.cvtColor(face_img_1, cv.COLOR_HSV2RGB_FULL)
        else:
            # erode that mask so the seam is inside the facial region.
            # this trick helps us remove the effect of strong gradient along seam
            final_mask = cv.morphologyEx(final_mask, cv.MORPH_ERODE, cv.getStructuringElement(cv.MORPH_RECT, (10,10)))
            face_img_1 = self._blend_images_opt(face_img, face_bgr, (final_mask==255).astype(np.float32), debug=False)

        # face_img_test = face_img.copy()
        # face_img_test[np.bitwise_not(final_mask==255)] = skin_color
        # plt.clf()
        # plt.subplot(131)
        # plt.imshow(face_img)
        # plt.subplot(132)
        # plt.imshow(face_img_test)
        # plt.subplot(133)
        # plt.imshow(face_img_1)
        # plt.savefig(f'/home/khanhhh/data_1/projects/Oh/data/face/google_front_faces/debug_cloning/{G_debug_id}_blend_input_img.png', dpi=500)
        return face_img_1

    def _fix_nostril_color(self, img, landmarks):
        """
        :param img: face image
        :param landmarks: 68x2 landamrks
        :return:
        """
        # segmentation
        dyn_seg = False
        if not dyn_seg:
            # two nostril centers share the same y coordinate
            l_ntr_center = 0.3*landmarks[30,:] + 0.7*landmarks[33,:]
            r_ntr_center = l_ntr_center.copy()
            # calculate x coordinate for two nostril centers
            l_ntr_center[0] = 0.5*(landmarks[31,0] + landmarks[32,0])
            r_ntr_center[0] = 0.5*(landmarks[34,0] + landmarks[35,0])
            l_ntr_center = l_ntr_center.astype(np.int32)
            r_ntr_center = r_ntr_center.astype(np.int32)
            radius = np.linalg.norm(landmarks[31,:] - landmarks[32,:])
            radius = int(radius)
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv.circle(mask, (l_ntr_center[0], l_ntr_center[1]), radius, (255,255,255), thickness=cv.FILLED)
            cv.circle(mask, (r_ntr_center[0], r_ntr_center[1]), radius, (255,255,255), thickness=cv.FILLED)

            # the mid region between two nostril
            # we also fill in this region for uniformity with two filled nostrils
            # other, the mid region still looks drak in some case while two nostrils look bright
            mid_idxs = [30, 32, 33, 34]
            mid_nostril_points = landmarks[mid_idxs,:].reshape((1,-1,2)).astype(np.int32)
            cv.fillPoly(mask, mid_nostril_points, (255,255,255))

            img_1 = cv.inpaint(img, mask, 3, cv.INPAINT_TELEA)
            #plt.axis('off')
            #plt.subplot(131)
            #plt.imshow(img)
            #plt.subplot(132)
            #plt.imshow(img)
            #plt.imshow(mask, alpha=0.4)
            #plt.subplot(133)
            #plt.imshow(img_1)
            #dir = "/home/khanhhh/data_1/projects/Oh/data/face/google_front_faces/debug_nostril/"
            #os.makedirs(dir, exist_ok=True)
            #plt.savefig(f'{dir}/{np.random.randint(1000)}.png', dpi=500)
            return img_1
        else:
            nostril_idxs = [30, 31, 33, 35]
            nostril_points = landmarks[nostril_idxs,:].astype(np.float32)
            nostril_points[2, :] = nostril_points[0,:] + 1.2 *(nostril_points[2,:] - nostril_points[0,:])
            nostril_points[1,:] = nostril_points[2,:] + 1.5*(nostril_points[1,:] - nostril_points[2,:])
            nostril_points[3,:] = nostril_points[2,:] + 1.5*(nostril_points[3,:] - nostril_points[2,:])
            nostril_points = nostril_points.astype(np.int32)

            ymin, ymax = nostril_points[:,0].min(), nostril_points[:,0].max()
            xmin, xmax = nostril_points[:,1].min(), nostril_points[:,1].max()

            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            mask[xmin:xmax, ymin:ymax] = 1
            mask = mask > 0
            model = BayesianGaussianMixture(n_components=3)
            cls_preds = model.fit_predict(img[mask])
            sorted_idxs = np.argsort(model.weights_)
            best_idx = sorted_idxs[0]
            best_colors_mask = cls_preds == best_idx
            mask_1 = mask.copy()
            mask_1[mask] = best_colors_mask
            mask_1 = mask_1.astype(np.uint8)*255
            mask_1 = cv.morphologyEx(mask_1, cv.MORPH_DILATE, cv.getStructuringElement(cv.MORPH_RECT, (5,5)))
            img_1 = cv.inpaint(img, mask_1, 3, cv.INPAINT_TELEA)

            # plt.subplot(141)
            # plt.imshow(img)
            # plt.imshow(mask, alpha=0.3)
            # plt.subplot(142)
            # plt.imshow(img)
            # plt.imshow(mask_1, alpha=0.3)
            # plt.subplot(143)
            # plt.imshow(img)
            # plt.subplot(144)
            # plt.imshow(img_1)
            #
            # dir = "/home/khanhhh/data_1/projects/Oh/data/face/google_front_faces/debug_nostril/"
            # os.makedirs(dir, exist_ok=True)
            # plt.savefig(f'{dir}/{np.random.randint(1000)}.png', dpi=500)
            #plt.show()
            return img_1

    def embed(self, prn_remap_tex, face_img, face_seg, face_landmarks, skin_color, fix_nostril = True, fix_hair_and_background=True):
        """

        :param prn_remap_tex: warping map from the face_img to PRN texture
        :param face_img: RGB image
        :param face_seg: segmentation map from the face parsing model
        :param face_landmarks:  68x2 landmarks
        :param fix_nostril: replace the black region below nostril with surrouding color or not
        :param fix_hair_and_background: replace hari and background region in the input face image with face color, so that they won't appear in the final texture
        :return:
        """
        if fix_hair_and_background:
            face_img = self._fix_hair_and_background(face_img, face_seg, skin_color)

        if fix_nostril:
            face_img = self._fix_nostril_color(face_img, face_landmarks)

        prn_facelib_tex = cv.remap(face_img, prn_remap_tex, None, interpolation=cv.INTER_AREA, borderMode=cv.BORDER_CONSTANT, borderValue=(0))

        assert prn_facelib_tex.shape[0] == prn_facelib_tex.shape[1], 'require square texture shape'

        face_texture = np.zeros(shape=(self.texture_size, self.texture_size, 3), dtype=np.uint8)
        face_texture[:,:,:] = skin_color

        if (2*self.embed_size + 1) != prn_facelib_tex.shape[0]:
            prn_facelib_tex = cv.resize(prn_facelib_tex, (2*self.embed_size + 1, 2*self.embed_size + 1), interpolation=cv.INTER_AREA)

        face_texture[self.rect_center[1]-self.embed_size:self.rect_center[1]+self.embed_size+1, self.rect_center[0]-self.embed_size:self.rect_center[0]+self.embed_size+1, :] = prn_facelib_tex

        head_texture = np.zeros((self.texture_size, self.texture_size, 3), dtype=np.uint8)
        head_texture[:,:,:] = skin_color

        #preprocess mask
        mask = self.face_tex_mask.astype(np.uint8) * 255
        head_texture_1 = self._blend_images_opt(face_texture, head_texture, (mask==255).astype(np.float32), debug=False)

        # debug
        # head_texture_2 = head_texture.copy()
        # head_texture_2[self.rect_center[1]-self.embed_size:self.rect_center[1]+self.embed_size+1, self.rect_center[0]-self.embed_size:self.rect_center[0]+self.embed_size+1, :] = prn_facelib_tex
        # plt.clf()
        # plt.subplot(221)
        # plt.imshow(face_texture)
        # plt.subplot(222)
        # plt.imshow(face_texture)
        # plt.imshow(mask, alpha=0.1)
        # plt.subplot(223)
        # plt.imshow(head_texture_2)
        # plt.subplot(224)
        # plt.imshow(head_texture_1)
        # plt.show()
        # plt.savefig(f'/home/khanhhh/data_1/projects/Oh/data/face/google_front_faces/debug_cloning/{G_debug_id}_blend_prn_tex.png', dpi=500)
        return head_texture_1

import pickle
if __name__ == '__main__':
    import matplotlib as mpl
    mpl.rc('image', cmap='gray')

    meta_data_dir = '/home/khanhhh/data_1/projects/Oh/codes/human_estimation/data/meta_data_shared/'
    model_dir = "/media/khanhhh/42855ff5-574e-4a41-ad10-0f08087b0ff6/data_1/projects/Oh/data/3d_human/deploy_models_nosyn"
    rect_path = os.path.join(*[meta_data_dir, 'prn_texture_in_victoria_texture.txt'])
    face_texture_processor = HmFPrnNetFaceTextureEmbedder(meta_dir=meta_data_dir, model_dir=model_dir, texture_size=1024)

    tmp_dir = '/home/khanhhh/data_1/projects/Oh/data/face/google_front_faces/tmp_data/'
    with open(f'{tmp_dir}/face_tmp_dat.pkl', 'rb') as file:
        data = pickle.load(file=file)
        prn_remap_tex = data['prn_remap_tex']
        img_face = data['img_face']
        img_face_landmarks = data['img_face_landmarks']
        texture = face_texture_processor.embed(prn_remap_tex, img_face)
