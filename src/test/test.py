import matplotlib.pyplot as plt
from pathlib import Path
import cv2 as cv
import numpy as np
from os.path import join
from sklearn.externals import joblib
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler, RobustScaler
from tqdm import tqdm
from common.obj_util import export_mesh, import_mesh
from os.path import join
from pca.nn_util import crop_silhouette_pair_blender
import tempfile

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

def analyze_dif_cam():
    dir_f = "/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj/caesar_silhouette_diff_cam/sil_f_fc"
    dir_s = "/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj/caesar_silhouette_diff_cam/sil_s_fc"
    names = sorted([path.name for path in Path(dir_f).glob('*.*')])

    size = (1024, 768)
    avg_sil_f = np.zeros(size, dtype=np.float)
    avg_sil_s = np.zeros(size, dtype=np.float)
    for idx, name in enumerate(names):
        #if idx != 0  and idx != len(names)-1:
        #      continue
        path_f = join(*[dir_f, name])
        path_s = join(*[dir_s, name])
        img_f = cv.imread(path_f, cv.IMREAD_GRAYSCALE)
        img_s = cv.imread(path_s, cv.IMREAD_GRAYSCALE)
        sil_f, sil_s = crop_silhouette_pair_blender(img_f, img_s, size=size)
        assert sil_f.max() == 255 and sil_s.max() == 255
        sil_f = sil_f.astype(np.float)/255.0
        sil_s = sil_s.astype(np.float)/255.0
        avg_sil_f += sil_f
        avg_sil_s += sil_s

    avg_sil_f /= float(len(names))
    avg_sil_s /= float(len(names))
    #avg_sil_f /= 2.0
    #avg_sil_s /= 2.0
    fig, axes = plt.subplots(1, 2, facecolor='red')
    axes[0].imshow(avg_sil_f)
    axes[1].imshow(avg_sil_s)
    fig.set_facecolor("white")
    plt.show()

def export_vic_pca_height():
    pca_dir = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj/pca_vic_coords/'
    model_path = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj/vic_pca_model.jlb'
    height_path = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj/height_syn.txt'
    pca_model = joblib.load(filename=model_path)

    paths = [path for path in Path(pca_dir).glob('*.npy')]
    heights = []
    for path in tqdm(paths):
        p = np.load(path)
        verts = pca_model.inverse_transform(p)
        verts = verts.reshape(verts.shape[0]//3, 3)
        h = verts[:,2].max() - verts[:,2].min()
        heights.append((path.stem, h))

    with open(height_path, 'wt') as file:
         file.writelines(f"{l[0]} {l[1]}\n" for l in heights)

def test_pca_max_min():
    pca_dir = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj/pca_vic_coords/'
    model_path = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj/vic_pca_model.jlb'
    height_path = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj/height_syn.txt'
    pca_model = joblib.load(filename=model_path)

    paths = [path for path in Path(pca_dir).glob('*.npy')]
    pca_vals = []
    for path in tqdm(paths):
        p = np.load(path)
        pca_vals.append(p)

    pca_vals = np.array(pca_vals)
    scale = RobustScaler()
    pca_vals_1 = scale.fit_transform(pca_vals)
    print(pca_vals.min(), pca_vals.max())
    print(pca_vals_1.min(), pca_vals_1.max())

def test_export_caesar_vic_mesh():
    vert_path = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj/victoria_caesar/CSR0097A.pkl'
    vic_mesh_path = '/home/khanhhh/data_1/projects/Oh/codes/human_estimation/data/meta_data/align_source_vic_mpii.obj'
    out_mesh_path = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj/victoria_caesar_obj/CSR0097A.obj'
    tpl_verts, tpl_faces = import_mesh(vic_mesh_path)
    verts = joblib.load(vert_path)
    export_mesh(fpath=out_mesh_path, verts=verts, faces=tpl_faces)

if __name__ == '__main__':

    #test_pca_max_min()
    #test_export_caesar_vic_mesh()
    analyze_dif_cam()