import matplotlib.pyplot as plt
from pathlib import Path
import cv2 as cv
import numpy as np
from os.path import join
def plot_silhouettes():
    dir_f = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj/cnn_data/sil_f_cropped/train'
    dir_s = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj/cnn_data/sil_s_cropped/train'
    dir_f = '/media/khanhhh/42855ff5-574e-4a41-ad10-0f08087b0ff6/data_1/projects/Oh/data/mobile_image_silhouettes/sil_f_cropped'
    dir_s = '/media/khanhhh/42855ff5-574e-4a41-ad10-0f08087b0ff6/data_1/projects/Oh/data/mobile_image_silhouettes/sil_s_cropped'
    n = 2

    paths_f = sorted([path for path in Path(dir_f).glob('*.*')])
    paths_s = sorted([path for path in Path(dir_s).glob('*.*')])

    idxs = np.random.randint(0, len(paths_f), n)
    paths_f = [paths_f[i] for i in idxs]
    paths_s = [paths_s[i] for i in idxs]

    fig, axes = plt.subplots(2, n)
    for i in range(n):
        img = cv.imread(str(paths_f[i]))
        axes[0, i].imshow(img)
    for i in range(n):
        img = cv.imread(str(paths_s[i]))
        axes[1, i].imshow(img)
    plt.show()

def load_img(path, color, bgr_color = (125,125,125)):
    img = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
    mask = img > 0
    img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    img[mask] = color
    img[np.bitwise_not(mask)] = bgr_color
    return img

def plot_diff_camera_sil():
    dir_f = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj/cnn_data/debug/CSR0309A/sil_f_cropped/'
    dir_s = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj/cnn_data/debug/CSR0309A/sil_s_cropped/'
    pathf_0 = join(*[dir_f, 'CSR0309A_dst_30.jpg'])
    pathf_1 = join(*[dir_f, 'CSR0309A_dst_50.jpg'])
    paths_0 = join(*[dir_s, 'CSR0309A_dst_30.jpg'])
    paths_1 = join(*[dir_s, 'CSR0309A_dst_50.jpg'])
    # pathf_0 = join(*[dir_f, 'CSR0309A_focal_len_2.6.jpg'])
    # pathf_1 = join(*[dir_f, 'CSR0309A_focal_len_4.4.jpg'])
    # paths_0 = join(*[dir_s, 'CSR0309A_focal_len_2.6.jpg'])
    # paths_1 = join(*[dir_s, 'CSR0309A_focal_len_4.4.jpg'])

    plt.rcParams['axes.facecolor'] = 'white'
    fig, axes = plt.subplots(1,2, facecolor='red')
    axes[0].imshow(load_img(pathf_0, (255,0,0)))
    axes[0].imshow(load_img(pathf_1, (0,0,255)), alpha=0.5)
    axes[1].imshow(load_img(paths_0, (255,0,0)))
    axes[1].imshow(load_img(paths_1, (0,0,255)), alpha=0.5)
    fig.set_facecolor("white")
    plt.show()

if __name__ == '__main__':

    plot_diff_camera_sil()

