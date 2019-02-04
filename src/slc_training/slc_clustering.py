import argparse
import pickle
import numpy as  np
from pathlib import Path
from src.common.util import load_bad_slice_names, calc_fourier_descriptor, reconstruct_contour_fourier, sample_contour_radial
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
import cv2 as cv
from sklearn.manifold import TSNE
import sklearn.metrics as metrics
import os
import shutil
from collections import defaultdict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def find_minimum_good_slc_names(SLC_DIR, BAD_SLC_DIR, ids):
    all_names = set()
    slc_names = []
    for id in ids:
        dir = f'{SLC_DIR}/{id}'
        bad_slc_names = load_bad_slice_names(BAD_SLC_DIR, id)

        names = set()
        for slc_path in Path(dir).glob('*.*'):
            if slc_path.stem not in bad_slc_names:
                names.add(slc_path.name)
                all_names.add(slc_path.name)

        slc_names.append(names)

    common_names = []
    for name in all_names:
        in_all = True
        for names in slc_names:
            if name not in names:
                in_all = False
                break

        if in_all == True:
            common_names.append(name)

    return common_names

def load_slc_w_d_ratio(IN_DIR, slc_id, names):
    ratios = []
    contours = []
    SLC_DIR = f'{IN_DIR}/{slc_id}'
    for name in names:
        slc_path = f'{SLC_DIR}/{name}'
        with open(str(slc_path), 'rb') as file:
            data = pickle.load(file=file)
            w = data['W']
            d = data['D']
            contour = data['cnt']
            if np.isclose(d, 0.0) or np.isclose(w, 0.0):
                raise RuntimeError('zero with or depth')
            ratios.append(w/d)
            contours.append(contour)

    return np.array(ratios).reshape(-1,1), contours

def convert_contour_fourier_code(contours, N):
    Fs = []
    for cnt in contours:
        F = calc_fourier_descriptor(cnt[0, :], cnt[1, :], N)
        Fs.append(F)
    Fs = np.array(Fs)
    return Fs

def apply_clustering(X, K):
    cluster_model = KMeans(n_clusters=K, n_init=50)
    labels = cluster_model.fit_predict(X)

    #score = metrics.silhouette_score(X, labels, metric='euclidean')
    #print(f'score = {score}')
    return labels

def plot_cluster_contour(labels, all_names, all_contours, DIR_OUT):
    clusters = {}
    unq_clusters = np.unique(labels)
    for k in unq_clusters:
        cluster_idxs = np.argwhere(labels == k)
        clusters[k] = cluster_idxs.flatten()

    shutil.rmtree(DIR_OUT, ignore_errors=True)
    os.makedirs(DIR_OUT, exist_ok=True)

    img_shape = (900, 700)
    img_center = np.array((int(img_shape[1]/2), int(img_shape[0]/2)))

    heat_map = np.zeros(img_shape, dtype=np.float)
    tmp_mask = np.zeros(img_shape, dtype=np.uint8)

    all_cluster_var = 0.0
    for label, idxs in clusters.items():

        cluster_radius = []

        heat_map[:] = 0

        for idx in idxs:
            name = all_names[idx]
            if name not in all_contours:
                continue

            data = all_contours[name]

            radial_data = data[0]
            cluster_radius.append(radial_data[:,2])

            contour = data[1].T

            contour = contour.astype(np.int)
            contour = contour + img_center

            n_point = contour.shape[0]
            tmp_mask[:] = 0
            for i in range(n_point):
                i_next = (i + 1) % n_point
                p_i = contour[i,:]
                p_n = contour[i_next, :]
                cv.line(tmp_mask, tuple(p_i), tuple(p_n), color=(255,255,255))

            tmp_mask = cv.morphologyEx(tmp_mask, cv.MORPH_DILATE, cv.getStructuringElement(shape=cv.MORPH_RECT, ksize=(5,5) ) )
            heat_map[tmp_mask > 0] += 1
            #plt.imshow(heat_map)
            #plt.show()

        cluster_radius = np.array(cluster_radius)
        cluster_radius_var = np.std(cluster_radius, axis=1)
        cluster_mean_var = np.mean(cluster_radius_var)

        all_cluster_var += cluster_mean_var
        max_count = 0.5*(len(idxs))

        heat_map[heat_map > max_count] = max_count

        bi_map = np.zeros(heat_map.shape, dtype=np.uint8)
        bi_map[heat_map > 0] = 255

        plt.clf()
        plt.axes().set_aspect(1.0)
        plt.subplot(121)
        plt.imshow(bi_map)
        plt.subplot(122)
        plt.imshow(heat_map)
        plt.title(f'cluster_id = {label}, n_contour in cluster = {len(idxs)}, \nmean_radius_variance = {cluster_mean_var:0.7f}', fontsize=6)
        plt.savefig(f'{DIR_OUT}/heat_map_{label}.png', dpi=500)

    mean_var = all_cluster_var/len(clusters.keys())
    return mean_var

def convert_to_radial_points(DIR_SLC, n_sample = 30, DIR_VIZ = None):
    rad_contours = {}
    for path in Path(DIR_SLC).glob('*.*'):
        with open(str(path), 'rb') as file:
            data = pickle.load(file=file)
            contour = data['cnt']

            F = calc_fourier_descriptor(contour[0, :], contour[1, :], 100)
            contour_1 = reconstruct_contour_fourier(F)

            try:
                contour_1 = 30000.0 * contour_1
                points = sample_contour_radial(contour_1[0, :], contour_1[1, :], center=np.array([0.0, 0.0]), n_sample=n_sample)
                contour_2 = np.array(points)
                radius = np.linalg.norm(contour_2, axis=1).reshape([-1,1])

                rad_contours[path.name] = [np.concatenate([contour_2, radius], axis=1), contour_1]

                if DIR_VIZ is not None:
                    plt.clf()
                    plt.axes().set_aspect(1.0)
                    plt.plot(contour_2[:, 0], contour_2[:, 1], 'ob')
                    plt.savefig(f'{DIR_VIZ}/{path.stem}.png')

            except Exception as exp:
                continue

    return rad_contours

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-input", type=str)
    ap.add_argument("-bad_slc", type=str)
    ap.add_argument("-debug_dir", type=str)

    args = ap.parse_args()
    IN_DIR = args.input
    BAD_SLC_DIR = args.bad_slc
    DEBUG_DIR = args.debug_dir

    os.makedirs(DEBUG_DIR, exist_ok=True)

    slc_ids = ['Bust', 'Waist', 'Hip']
    names = find_minimum_good_slc_names(IN_DIR, BAD_SLC_DIR, slc_ids)

    tmp_path = f'{DEBUG_DIR}/contour_radial.pkl'
    if not Path(tmp_path).exists():
        print('calculating radial contour representation')
        slc_rad_contours = {}
        for slc_id in slc_ids:
            print(f'\tfor slice {slc_id}')
            SLC_DIR = f'{IN_DIR}/{slc_id}/'
            slc_rad_contours[slc_id] = convert_to_radial_points(SLC_DIR, DIR_VIZ=None)

        with open(tmp_path, 'wb') as file:
            pickle.dump(obj=slc_rad_contours, file=file)
    else:
        print(f'loading radius format from file {tmp_path}')
        with open(tmp_path, 'rb') as file:
            slc_rad_contours = pickle.load(file=file)

    bust_ratios, bust_contours  = load_slc_w_d_ratio(IN_DIR, 'Bust', names)
    waist_ratios, waist_contours = load_slc_w_d_ratio(IN_DIR, 'Waist', names)
    hip_ratios, hip_contours  = load_slc_w_d_ratio(IN_DIR, 'Hip', names)
    X = np.concatenate([bust_ratios, waist_ratios, hip_ratios], axis=1)
    print(X.shape)

    K = 8
    for slc_id in ['Hip', 'Bust', 'Waist']:
        print(f'\nstart clustering slice : {slc_id}')
        labels = apply_clustering(X, K=K)
        mean_var = plot_cluster_contour(labels, names, slc_rad_contours[slc_id], f'{DEBUG_DIR}/triple/{slc_id}')
        print(f'\ttripple model. radius var = {mean_var}')

        labels_1 = apply_clustering(bust_ratios, K=K)
        mean_var = plot_cluster_contour(labels_1, names, slc_rad_contours[slc_id], f'{DEBUG_DIR}/single/{slc_id}')
        print(f'\tsingle model. radius var = {mean_var}')

        hip_Fs = convert_contour_fourier_code(hip_contours, N=8)
        #hip_Fs = TSNE(n_components=3).fit_transform(hip_Fs)
        labels_2 = apply_clustering(hip_Fs, K=K)
        mean_var =plot_cluster_contour(labels_2, names, slc_rad_contours[slc_id], f'{DEBUG_DIR}/F_code/{slc_id}')
        print(f'\tfourier model. radius var = {mean_var}')

    # plt.clf()
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(X_embedded[:,0], X_embedded[:,1], X_embedded[:,2], marker='o')
    # plt.show()
    exit()
