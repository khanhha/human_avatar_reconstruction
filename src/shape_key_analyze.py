import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path


def pre_process(DIR_IN, DIR_OUT):

    for img_path in Path(DIR_OUT).glob("*.*"):
        os.remove(img_path)

    for img_path in Path(DIR_IN).glob("*.*"):
        img = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)
        sil_mask = img > 0
        sil = (sil_mask * 255).astype(np.uint8)
        sil = cv.morphologyEx(sil, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (5,5)))
        cv.imwrite(f'{DIR_OUT}{img_path.name}', sil)

def viz_shapekey_silhouette_dif(DIR_IN, DIR_OUT):
    for img_path in Path(DIR_OUT).glob("*.*"):
        os.remove(img_path)
    shapekey_paths = {}
    for img_path in Path(DIR_IN).glob("*.png"):
        name_id = str(img_path.name).split('_')[0]
        if name_id not in shapekey_paths:
            shapekey_paths[name_id] = []
        shapekey_paths[name_id].append(img_path)

    sil_area = -1

    for id, paths in shapekey_paths.items():
        for img_path in  paths:
            if 'front_0' in img_path.name:
                img_front_0 = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)
            elif 'front_1' in img_path.name:
                img_front_1 = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)
            elif 'side_0' in img_path.name:
                img_side_0 = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)
            elif 'side_1' in img_path.name:
                img_side_1 = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)
            else:
                print('error: unexpected filename')

        if sil_area == -1:
            sil_area = float(np.sum((img_front_0>0)[:]))

        dif_front = cv.bitwise_xor(img_front_0, img_front_1)
        dif_side  = cv.bitwise_xor(img_side_0, img_side_1)

        front_change_area = np.sum((dif_front > 0)[:])
        side_change_area = np.sum((dif_side > 0)[:])

        if float(front_change_area)/sil_area > 0.005 or float(side_change_area)/sil_area > 0.005:
            dif_front = np.dstack([dif_front, np.zeros_like(dif_front), np.zeros_like(dif_front)])
            dif_side = np.dstack([dif_side, np.zeros_like(dif_side), np.zeros_like(dif_side)])

            plt.clf()
            plt.subplot(121)
            plt.imshow(img_front_0[:,600:1300], cmap='gray')
            plt.imshow(dif_front[:,600:1300], alpha=0.5)
            plt.subplot(122)
            plt.imshow(img_side_0[:,600:1300], cmap='gray')
            plt.imshow(dif_side[:,600:1300], alpha=0.5)
            plt.savefig(f'{DIR_OUT}{id}.png', dpi=500)

if __name__  == '__main__':
    BLENDER_RENDER_DIR = '/home/khanhhh/data_1/projects/Oh/data/bl_models/blender_silhouette/'
    OUT_SILHOUETTE_DIR = '/home/khanhhh/data_1/projects/Oh/data/bl_models/silhouette_shapekey/'
    OUT_SIL_DIF_DIR = '/home/khanhhh/data_1/projects/Oh/data/bl_models/silhouette_shapekey_analysis/'
    pre_process(BLENDER_RENDER_DIR, OUT_SILHOUETTE_DIR)
    viz_shapekey_silhouette_dif(OUT_SILHOUETTE_DIR, OUT_SIL_DIF_DIR)

