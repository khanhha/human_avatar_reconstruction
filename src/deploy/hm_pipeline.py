from deploy.hm_shape_pred_pytorch_model import HmShapePredPytorchModel
from deploy.hm_shape_pred_model import HmShapePredModel
from deploy.hm_sil_pred_model import HmSilPredModel
from pca.nn_util import crop_silhouette_pair
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from pathlib import Path
from common.obj_util import import_mesh_obj, export_mesh
from deploy.data_config import  config_get_data_path
from pose.pose_extract_tfpose import PoseExtractorTf
#this module predicts a 3D human mesh from front, side images, height and gender
class HumanRGBModel:

    def __init__(self, hmshape_model_path, hmsil_model_path, mesh_path):
        self.hmsil_model = HmSilPredModel(model_path=hmsil_model_path, use_gpu=True, use_mobile_model=False)
        #self.hmshape_model = HmShapePredPytorchModel(model_path=hmshape_model_path)
        self.hmshape_model = HmShapePredModel(model_path=hmshape_model_path)
        _, faces = import_mesh_obj(mesh_path)

        self.extractor = PoseExtractorTf()
        self.tpl_faces = faces

    def predict(self, rgb_img_f, rgb_img_s, height, gender, correct_silhouette = False):
        """
        predict a 3D human mesh from front side RGB images, height and gender
        :param rgb_img_f:
        :param rgb_img_s:
        :param height: in meter, for exp 1.6
        :param gender: 0 for female and 1 for male
        :return:
        # verts: NX3 points
        # faces: template face list
        # sil_f: front silhouette
        # sil_s: side silhouette
        """
        sil_f = self.hmsil_model.extract_silhouette(rgb_img_f)
        sil_s = self.hmsil_model.extract_silhouette(rgb_img_s)
        sil_f = sil_f.astype(np.float32)/255.0
        sil_s = sil_s.astype(np.float32)/255.0
        # fig, axes = plt.subplots(1,2)
        # axes[0].imshow(rgb_img_f)
        # axes[0].imshow(sil_f, alpha=0.5)
        # axes[1].imshow(rgb_img_s)
        # axes[1].imshow(sil_s, alpha=0.5)
        # plt.show()
        pose_f, pose_s = None, None
        if correct_silhouette:
            pose_f, img_pose_f = self.extractor.extract_single_pose(rgb_img_f, debug=True)
            pose_s, img_pose_s = self.extractor.extract_single_pose(rgb_img_s, debug=True)
            # plt.subplot(121)
            # plt.imshow(img_pose_f)
            # plt.subplot(122)
            # plt.imshow(img_pose_s)
            # plt.show()

        verts, faces  = self.predict_sil(sil_f, sil_s, height, gender, pose_f=pose_f, pose_s=pose_s)

        return verts, faces, sil_f, sil_s

    def predict_sil(self, sil_f, sil_s, height, gender, pose_f = None, pose_s = None):
        """
        :param sil_f: front silhouete
        :param sil_s: side silhouete
        :param height:  in meter
        :param gender:  0 for female and 1 for male
        :return:
        verts: Nx3 points
        faces: template face list
        """
        verts = self.hmshape_model.predict(sil_f=sil_f, sil_s=sil_s, height=height, gender=gender, pose_f=pose_f, pose_s=pose_s)
        verts = verts[0]
        verts = verts.reshape(verts.shape[0]//3, 3)
        return verts, self.tpl_faces

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-model_dir",  required=True, type=str, help="the direction where shape_model.jlb  and deeplab model are stored")
    ap.add_argument("-in_txt_file",  required=True, type=str, help="the texture file to the input data directory. each line of the texture file have the following format:"
                                                                   "front_img  side_img  height_in_meter  gender  are_silhouette_or_not")
    ap.add_argument("-save_sil",  required=False, default= True, type=bool, help="save silhouettes that are calculated from RGB input images")

    args = ap.parse_args()

    shape_model_path = config_get_data_path(args.model_dir, 'shape_model')
    deeplab_path = config_get_data_path(args.model_dir, 'deeplab_tensorflow_model')
    vic_mesh_path  = config_get_data_path(args.model_dir, 'victoria_template_mesh')

    assert Path(shape_model_path).exists() and Path(deeplab_path).exists()
    model = HumanRGBModel(hmshape_model_path=shape_model_path, hmsil_model_path=deeplab_path, mesh_path=vic_mesh_path)

    with open(args.in_txt_file, 'rt') as file:
        dir = Path(args.in_txt_file).parent

        for idx, l in enumerate(file.readlines()):
            if l[0] == '#':
                continue
            comps = l.split(' ')
            front_img_path = os.path.join(*[dir, comps[0]])
            side_img_path  = os.path.join(*[dir, comps[1]])
            height = float(comps[2])
            gender = float(comps[3])
            is_sil = int(comps[4])

            assert gender == 0.0 or gender == 1.0 , 'unexpected gender. just accept 1 or 0'
            assert is_sil == 0 or is_sil == 1, 'unexpected sil flag. just accept 1 or 0'
            assert height >= 1.0 and height  <= 2.5, 'unexpected height. not in range of [1.0, 2.5]'

            assert Path(front_img_path).exists() and Path(side_img_path).exists()

            print(f'\nprocess image pair {comps[0]} - {comps[1]}. height = {height}. gender = {gender}. is_sil = {is_sil}')
            if not is_sil:
                img_f = cv.imread(front_img_path)
                img_s = cv.imread(side_img_path)

                verts, faces, sil_f, sil_s = model.predict(img_f, img_s, height, gender)

                if args.save_sil:
                    sil_f  = (sil_f*255.0).astype(np.uint8)
                    sil_s  = (sil_s*255.0).astype(np.uint8)
                    out_sil_f_path = os.path.join(*[dir, f'{Path(front_img_path).stem}_sil.jpg'])
                    cv.imwrite(out_sil_f_path, sil_f)
                    out_sil_s_path = os.path.join(*[dir, f'{Path(side_img_path).stem}_sil.jpg'])
                    cv.imwrite(out_sil_s_path, sil_s)
                    print(f'\texported silhouette {out_sil_f_path} - {out_sil_s_path}')
            else:
                img_f = cv.imread(front_img_path, cv.IMREAD_GRAYSCALE)
                img_s = cv.imread(side_img_path, cv.IMREAD_GRAYSCALE)
                ret_0, img_f = cv.threshold(img_f, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
                ret_1, img_s = cv.threshold(img_s, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
                # plt.subplot(121)
                # plt.imshow(img_f)
                # plt.subplot(122)
                # plt.imshow(img_s)
                # plt.show()
                verts, faces = model.predict_sil(img_f, img_s, height, gender)

            out_path = os.path.join(*[dir, f'{Path(front_img_path).stem}_{idx}.obj'])
            print(f'\texported obj object to path {out_path}')
            export_mesh(out_path, verts=verts, faces=faces)

