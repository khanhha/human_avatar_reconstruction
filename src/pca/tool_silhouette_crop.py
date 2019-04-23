import argparse
import os
from os.path import join
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2 as cv
from pca.nn_util import crop_silhouette_pair_blender
import numpy as np

def find_largest_contour(img_bi, app_type=cv.CHAIN_APPROX_TC89_L1):
    contours, _ = cv.findContours(img_bi, cv.RETR_LIST, app_type)
    largest_cnt = None
    largest_area = -1
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > largest_area:
            largest_area = area
            largest_cnt = cnt
    return largest_cnt

def draw_silhouette(draw_img, sil, color = (255,0,0)):
    contour = find_largest_contour(sil, app_type=cv.CHAIN_APPROX_NONE)
    cv.drawContours(draw_img, [contour], -1, thickness=1, color=color)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-sil_f_dir", type=str, required=True)
    ap.add_argument("-sil_s_dir", type=str, required=True)
    ap.add_argument("-out_dir",   type=str, required=True)
    ap.add_argument("-resize_size",   type=str, required=False, default='224x224')
    args = ap.parse_args()

    size = args.resize_size.split('x')
    size = (int(size[0]), int(size[1]))

    sil_f_dir_out = join(*[args.out_dir, 'sil_f'])
    sil_s_dir_out = join(*[args.out_dir, 'sil_s'])
    os.makedirs(sil_f_dir_out, exist_ok=True)
    os.makedirs(sil_s_dir_out, exist_ok=True)

    for path in Path(sil_f_dir_out).glob('*.*'):
        os.remove(str(path))
    for path in Path(sil_s_dir_out).glob('*.*'):
        os.remove(str(path))

    fnames = [path.name for path in Path(args.sil_f_dir).glob('*.*')]

    sil_pairs = []
    for name in tqdm(fnames):
        f_path = join(*[args.sil_f_dir, name])
        s_path = join(*[args.sil_s_dir, name])

        sil_f_org = cv.imread(str(f_path), cv.IMREAD_GRAYSCALE)
        sil_s_org = cv.imread(str(s_path), cv.IMREAD_GRAYSCALE)

        sil_f_1, sil_s_1 = crop_silhouette_pair_blender(sil_f_org, sil_s_org, size)

        #plt.subplot(121), plt.imshow(sil_f_org)
        #plt.subplot(122), plt.imshow(sil_s_org)
        #plt.show()

        f_path_out = join(*[sil_f_dir_out, f'{name[:-4]}.jpg'])
        s_path_out = join(*[sil_s_dir_out, f'{name[:-4]}.jpg'])
        #print(f'output {f_path_out}, {s_path_out}')
        cv.imwrite(img=sil_f_1, filename=str(f_path_out))
        cv.imwrite(img=sil_s_1, filename=str(s_path_out))

