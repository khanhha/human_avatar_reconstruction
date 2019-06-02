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

class HumanRGBModel:

    def __init__(self, hmshape_model_path, hmsil_model_path, mesh_path):
        self.hmsil_model = HmSilPredModel(model_path=hmsil_model_path, use_gpu=True, use_mobile_model=False)
        #self.hmshape_model = HmShapePredPytorchModel(model_path=hmshape_model_path)
        self.hmshape_model = HmShapePredModel(model_path=hmshape_model_path)
        _, faces = import_mesh_obj(mesh_path)
        self.tpl_faces = faces

    def predict(self, rgb_img_f, rgb_img_s, height, gender):
        sil_f = self.hmsil_model.extract_silhouette(rgb_img_f)
        sil_s = self.hmsil_model.extract_silhouette(rgb_img_s)
        sil_f = sil_f.astype(np.float32)/255.0
        sil_s = sil_s.astype(np.float32)/255.0
        fig, axes = plt.subplots(1,2)
        # axes[0].imshow(sil_f)
        # axes[1].imshow(sil_s)
        # plt.show()

        verts = self.hmshape_model.predict(sil_f=sil_f, sil_s=sil_s, height=height, gender=gender)
        verts = verts[0]
        verts = verts.reshape(verts.shape[0]//3, 3)
        return verts, self.tpl_faces

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-model_dir",  required=True, type=str, help="the direction where shape_model.jlb  and deeplab model are stored")
    ap.add_argument("-img_f",  required=True, type=str, help="path to the front image")
    ap.add_argument("-img_s",  required=True, type=str, help="path to the side image")
    ap.add_argument("-out_obj_path",  required=True, type=str, help="the obj ouput path")
    ap.add_argument("-height",  required=True, type=float, help="height of that person, in meter, for example, 1.62")
    ap.add_argument("-gender",  required=True, type=int, choices=[0, 1], help="gender of that person: 1 is male and 0 is male")

    args = ap.parse_args()

    #model_dir = '/home/khanhhh/data_1/projects/Oh/data/3d_human/deploy_models/'
    #img_f_path = '/home/khanhhh/data_1/projects/Oh/data/oh_mobile_images/images/front_IMG_1928.JPG'
    #img_s_path = '/home/khanhhh/data_1/projects/Oh/data/oh_mobile_images/images/side_IMG_1938.JPG'

    #shape_model_path = os.path.join(*[args.model_dir, 'shape_model_pytorch.pt'])
    shape_model_path = os.path.join(*[args.model_dir, 'shape_model.jlb'])
    deeplab_path = os.path.join(*[args.model_dir, 'deeplabv3_xception_ade20k_train_2018_05_29.tar.gz'])
    vic_mesh_path = os.path.join(*[args.model_dir, 'vic_mesh.obj'])

    assert Path(shape_model_path).exists() and Path(deeplab_path).exists() and Path(args.img_f).exists() and Path(args.img_s).exists()
    model = HumanRGBModel(hmshape_model_path=shape_model_path, hmsil_model_path=deeplab_path, mesh_path=vic_mesh_path)

    img_f = cv.imread(args.img_f)
    img_s = cv.imread(args.img_s)

    height = args.height
    gender = float(args.gender)

    verts, faces = model.predict(img_f, img_s, height, gender)

    print(f'output mesh shape: {verts.shape}')
    print(f'exported obj object to path {args.out_obj_path}')
    export_mesh(args.out_obj_path, verts=verts, faces=faces)

