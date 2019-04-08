import argparse
import os
from os.path import join
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2 as cv

def aspect(img):
    return float(img.shape[0]) / img.shape[1]

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-dir", type=str, required=True)
    ap.add_argument("-tpl_front_sil", type=str, required=True)
    ap.add_argument("-tpl_side_sil",  type=str, required=True)
    args = ap.parse_args()

    sil_f_dir = join(*[args.dir, 'sil_f_org'])
    sil_s_dir = join(*[args.dir, 'sil_s_org'])

    sil_f_dir_out = join(*[args.dir, 'sil_f'])
    sil_s_dir_out = join(*[args.dir, 'sil_s'])

    for path in Path(sil_f_dir_out).glob('*.*'):
        os.remove(str(path))
    for path in Path(sil_s_dir_out).glob('*.*'):
        os.remove(str(path))

    tpl_sil_f = cv.cvtColor(cv.imread(str(args.tpl_front_sil)), cv.COLOR_BGR2GRAY)
    tpl_sil_s = cv.cvtColor(cv.imread(str(args.tpl_side_sil)), cv.COLOR_BGR2GRAY)

    asp_sil_f_tpl = aspect(tpl_sil_f)
    asp_sil_s_tpl = aspect(tpl_sil_s)

    fnames = [path.name for path in Path(sil_f_dir).glob('*.*')]

    for name in fnames:
        f_path = join(*[sil_f_dir, name])
        s_path = join(*[sil_s_dir, name])
        sil_f = cv.cvtColor(cv.imread(str(f_path)), cv.COLOR_BGR2GRAY)
        sil_s = cv.cvtColor(cv.imread(str(s_path)), cv.COLOR_BGR2GRAY)


        for ver_ext in range(100, 500, 5):
            for hor_ext in range(100, 500, 5):
                sil_f_1 = cv.copyMakeBorder(sil_f, top=ver_ext, bottom=ver_ext, left=hor_ext, right=hor_ext, borderType=cv.BORDER_CONSTANT)
                sil_s_1 = cv.copyMakeBorder(sil_s, top=ver_ext, bottom=ver_ext, left=hor_ext, right=hor_ext, borderType=cv.BORDER_CONSTANT)

                asp_sil_f = aspect(sil_f_1)
                asp_sil_s = aspect(sil_s_1)

                if abs(asp_sil_f-asp_sil_f_tpl) > 0.001 or abs(asp_sil_s - asp_sil_s_tpl) > 0.001:
                    continue

                print(f'picked padding : {ver_ext}, {hor_ext}')

                # plt.clf()
                # plt.subplot(221)
                # plt.imshow(sil_f)
                # plt.subplot(222)
                # plt.imshow(sil_s)
                #
                # plt.subplot(223)
                # plt.imshow(sil_f_1)
                # plt.subplot(224)
                # plt.imshow(sil_s_1)
                #
                # plt.show()

                f_path_out = join(*[sil_f_dir_out, f'{name[:-4]}_{ver_ext}_{hor_ext}.jpg'])
                s_path_out = join(*[sil_s_dir_out, f'{name[:-4]}_{ver_ext}_{hor_ext}.jpg'])
                cv.imwrite(img=sil_f_1, filename=str(f_path_out))
                cv.imwrite(img=sil_s_1, filename=str(s_path_out))

