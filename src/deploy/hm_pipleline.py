from deploy.hm_shape_pred_model import HmShapePredModel
from deploy.hm_sil_pred_model import HmSilPredModel
from pca.nn_util import crop_silhouette_pair
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

class HumanRGBModel:

    def __init__(self, hmshape_model_path, hmsil_model_path):
        self.hmsil_model = HmSilPredModel(use_gpu=True, use_mobile_model=False, save_model_path=hmsil_model_path)
        self.hmshape_model = HmShapePredModel(hmshape_model_path)

    def predict(self, rgb_img_f, rgb_img_s, height, gender):
        sil_f = self.hmsil_model.extract_silhouette(rgb_img_f)
        sil_s = self.hmsil_model.extract_silhouette(rgb_img_s)
        #fig, axes = plt.subplots(1,2)
        # axes[0].imshow(sil_f)
        # axes[1].imshow(sil_s)
        # plt.show()
        verts = self.hmshape_model.predict(sil_f=sil_f, sil_s=sil_s, height=height, gender=gender)
        return verts

if __name__ == '__main__':
    shape_path = '/home/khanhhh/data_1/projects/Oh/data/3d_human/deploy_models/shape_model.jlb'
    sil_path = '/home/khanhhh/data_1/projects/Oh/data/3d_human/deploy_models/'

    img_f_path = '/home/khanhhh/data_1/projects/Oh/data/oh_mobile_images/images/front_IMG_1928.JPG'
    img_s_path = '/home/khanhhh/data_1/projects/Oh/data/oh_mobile_images/images/side_IMG_1938.JPG'

    model = HumanRGBModel(hmshape_model_path=shape_path, hmsil_model_path=sil_path)

    img_f = cv.imread(img_f_path)
    img_s = cv.imread(img_s_path)

    height = 1.52
    gender = 0.0

    verts = model.predict(img_f, img_s, height, gender)

    print(f'output mesh shape: {verts.shape}')

