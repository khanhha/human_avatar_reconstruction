import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np
from os.path import join
import os
import shutil
import cv2 as cv
from pca.nn_util import  crop_silhouette_pair, verify_pose_variants_per_name, remove_pose_variant_in_file_name
from tqdm import tqdm
import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import partial
import tempfile
from pca.pca_vic_model import PcaModel
from sklearn.externals import joblib
import sklearn
from common.util import find_largest_contour, smooth_contour, resample_contour

label_colors = {'head':(128,0,0), 'torso':(255,85,0),
                'larm':(51,170,221), 'rarm':(0,255,255),
                'lleg_upper':(0,85,85,0), 'rleg_upper':(0,150,25),
                'lleg_lower':(85,255,170), 'rleg_lower':(170,255,85),
                'lfoot':(255,255,0), 'rfoot':(255,170,0)}

def body_part_mask(rgb_img, rgb_value, rgb_epsilon):
    if isinstance(rgb_epsilon, int):
        rgb_epsilon = (rgb_epsilon, rgb_epsilon, rgb_epsilon)

    r_mask = np.bitwise_and(rgb_img[:,:,0] > rgb_value[0]-rgb_epsilon[0], rgb_img[:,:,0] < rgb_value[0]+rgb_epsilon[0])
    g_mask = np.bitwise_and(rgb_img[:,:,1] > rgb_value[1]-rgb_epsilon[1], rgb_img[:,:,1] < rgb_value[1]+rgb_epsilon[1])
    b_mask = np.bitwise_and(rgb_img[:,:,2] > rgb_value[2]-rgb_epsilon[2], rgb_img[:,:,2] < rgb_value[2]+rgb_epsilon[2])
    mask = np.bitwise_and(r_mask, np.bitwise_and(g_mask, b_mask))
    return (mask*255).astype(np.uint8)

def extract_body_part_masks(img_rgb):
    body_masks = {}
    for k, v in label_colors.items():
        mask = body_part_mask(img_rgb, v, 10)
        body_masks[k] = mask
    return body_masks

def segment_body_part_sil_f(path):
    img = cv.imread(str(path))
    img = img[400:, :, ::-1]

    masks = extract_body_part_masks(img)
    img_1 = img.copy()
    contours = []
    for k, mask in masks.items():
        contour = find_largest_contour(mask, app_type=cv.CHAIN_APPROX_NONE)
        X, Y = smooth_contour(contour[:,0,0], contour[:,0,1], sigma=4)
        contour_1 = np.vstack([X,Y]).T
        contours.append(contour_1)
        cv.fillConvexPoly(img_1, contour_1.reshape(-1,1,2), color=label_colors[k])

    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(img_1)
    plt.show()

def copy_files(paths, out_dir, args):
    for path in Path(out_dir).glob('*.*'):
        os.remove(str(path))

    os.makedirs(out_dir, exist_ok=True)
    for path in tqdm(paths, desc=f'copied files to {out_dir}'):
        out_name = str(path.stem).replace('_front','')
        out_name = out_name.replace('_side', '')
        shutil.copy(src=path, dst=join(*[out_dir, f'{out_name}.jpg']))

def copy_train_valid_test(name_ids, id_to_path_dict, train_idxs, valid_idxs, test_idxs, out_dir, args):
    train_dir = join(*[out_dir, 'train'])
    valid_dir = join(*[out_dir, 'valid'])
    test_dir  = join(*[out_dir, 'test'])

    copy_files([id_to_path_dict[name_ids[idx]] for idx in train_idxs], train_dir, args)
    copy_files([id_to_path_dict[name_ids[idx]] for idx in valid_idxs], valid_dir, args)
    copy_files([id_to_path_dict[name_ids[idx]] for idx in test_idxs], test_dir, args)

def extract_silhouette(img, background_I, epsilon):
    masks = []
    for i in range(3):
        m = np.bitwise_and(img[:,:,i] > background_I - epsilon, img[:, :, i] < background_I + epsilon)
        masks.append(m)

    mask = np.bitwise_and(np.bitwise_and(masks[0], masks[1]), masks[2])
    return np.bitwise_not(mask).astype(np.uint8)*255

def crop_a_pair(size, path_pair):
    fpath = path_pair[0]
    spath = path_pair[1]

    sil_f = cv.imread(str(fpath))
    sil_s = cv.imread(str(spath))

    assert sil_f is not None, f'{fpath} image does not exist'
    assert sil_s is not None, f'{spath} image does not exist'

    background_intensity = 54
    epsilon = 5

    sil_f = extract_silhouette(sil_f, background_intensity, epsilon)
    sil_s = extract_silhouette(sil_s, background_intensity, epsilon)

   # plt.subplot(121), plt.imshow(sil_f)
   # plt.subplot(122), plt.imshow(sil_s)
   # plt.show()

    sil_f, sil_s, _, _ = crop_silhouette_pair(sil_f, sil_s, mask_f=sil_f, mask_s=sil_s, target_h=size[0], target_w=size[1], px_height=int(0.9 * size[0]))
    #sil_f, sil_s = crop_silhouette_pair_blender(sil_f, sil_s, size)

    #plt.subplot(121), plt.imshow(sil_f)
    #plt.subplot(122), plt.imshow(sil_s)
    #plt.show()

    cv.imwrite(str(fpath), img=sil_f)
    cv.imwrite(str(spath), img=sil_s)

def crop_pairs(sil_f_dir, sil_s_dir, size):
    fpaths = sorted([path for path in Path(sil_f_dir).glob('*.*')])
    spaths = sorted([path for path in Path(sil_s_dir).glob('*.*')])
    for fpath, spath in zip(fpaths, spaths):
        assert fpath.name == spath.name

    path_pairs = [(fpath, spath) for fpath, spath in zip(fpaths, spaths)]

    with Pool(1) as p:
        with tqdm(total=len(path_pairs), desc=f'cropping pair: {Path(sil_f_dir).stem}, {Path(sil_s_dir).stem}') as pbar:
            for i, _ in enumerate(p.imap_unordered(partial(crop_a_pair, size), path_pairs)):
                pbar.update()

def crop_train_test_valid(base_sil_f_dir, base_sil_s_dir, size):
    names = ['train', 'test', 'valid']
    for name in names:
        f_dir = join(*[base_sil_f_dir, name])
        s_dir = join(*[base_sil_s_dir, name])
        crop_pairs(f_dir, s_dir, size)

def copy_file_prefix(in_dir, out_dir, prefix, n_file = -1):
    os.makedirs(out_dir, exist_ok=True)

    paths = list([path for path in Path(in_dir).glob('*.*')])
    paths = sorted(paths)
    #paths = sklearn.utils.shuffle(paths)
    if n_file > 0:
        paths = paths[:n_file]

    for path in tqdm(paths, desc= f'copy_file_prefix {prefix}'):
        shutil.copy(str(path), os.path.join(*[out_dir, f'{prefix}_{path.name}']))

def copy_target_prefix(in_dir, out_dir, prefix, pose_duplicate = 0, n_files = -1):
    assert prefix in ['_male', '_female']
    os.makedirs(out_dir, exist_ok=True)

    ex_val = np.array([1.0]) if prefix == '_male' else np.array([0.0])

    paths = sorted([path for path in Path(in_dir).glob('*.*')])
    if n_files > 0:
        paths = paths[:n_files]

    for path in tqdm(paths, desc=f'copy_target_prefix {prefix}'):
        param = np.load(path)
        param = np.hstack([ex_val, param])
        if pose_duplicate == 0:
            np.save(os.path.join(*[out_dir, f'{prefix}_{path.name}']), param)
        else:
            for i in range(pose_duplicate):
                np.save(os.path.join(*[out_dir, f'{prefix}_{path.stem}_pose{i}{path.suffix}']), param)

def remove_missing_pair(sil_f_dir, sil_s_dir):
    sil_f_names = set([path.name for path in Path(sil_f_dir).glob('*.*')])
    sil_s_names = set([path.name for path in Path(sil_s_dir).glob('*.*')])

    common_names = sil_f_names.intersection(sil_s_names)

    bad_f_names = sil_f_names.difference(common_names)
    bad_s_names = sil_s_names.difference(common_names)

    for name in bad_f_names:
        print(f'remove file {name}')
        path = os.path.join(*[sil_f_dir, name])
        os.remove(path)

    for name in bad_s_names:
        print(f'remove file {name}')
        path = os.path.join(*[sil_s_dir, name])
        os.remove(path)

def verify_missing_pair(sil_f_dir, sil_s_dir):
    sil_f_names = [path.name for path in Path(sil_f_dir).glob('*.*')]
    sil_s_names = [path.name for path in Path(sil_s_dir).glob('*.*')]
    sil_f_names = sorted(sil_f_names)
    sil_s_names = sorted(sil_s_names)

    for f_name, s_name in zip(sil_f_names, sil_s_names):
        assert f_name == s_name, 'missing pair'

def dump_heights(pca_in_dir, pca_ml_model_path, pca_fml_model_path, height_out_path, n_files = -1):
    ml_model = joblib.load(pca_ml_model_path)
    fml_model = joblib.load(pca_fml_model_path)

    paths = sorted([path for path in Path(pca_in_dir).glob('*.*')])
    if n_files > 0:
        paths = paths[:n_files]

    heights = []
    for path in tqdm(paths, desc='dump height'):
        param = np.load(path)
        if '_male' in path.stem:
            assert param[0] >= 0.9, f'{param[0]} >= 0.9, {path.name}'
            verts = ml_model.inverse_transform(param[1:])
        else:
            assert param[0] <= 0.1, f'{param[0]} <= 0.1, {path.name}'
            verts = fml_model.inverse_transform(param[1:])

        verts = verts.reshape(verts.shape[0] // 3, 3)
        h = verts[:, 2].max() - verts[:, 2].min()
        heights.append((path.stem, h))

    with open(height_out_path, 'wt') as file:
        file.writelines(f"{l[0]} {l[1]}\n" for l in heights)


def verify_splitting_pose_variant(paths, train_idxs, valid_idxs, test_idxs, N_pose_per_subject):
    """
    verify that all pose variants of a subject stay completely inside each set
    :param paths:
    :param train_idxs:
    :param valid_idxs:
    :param test_idxs:
    :param N_pose_per_subject:
    """
    train_paths = [paths[idx] for idx in train_idxs]
    valid_paths = [paths[idx] for idx in valid_idxs]
    test_paths =  [paths[idx] for idx in test_idxs]
    verify_pose_variants_per_name(train_paths, N_pose_per_subject)
    verify_pose_variants_per_name(valid_paths, N_pose_per_subject)
    verify_pose_variants_per_name(test_paths, N_pose_per_subject)

def split_train_valid_test_pose_variants(sil_f_paths, sil_s_paths):
    assert len(sil_f_paths) == len(sil_s_paths)
    for path_f, path_s in zip(sil_f_paths, sil_s_paths):
        assert Path(path_f).name == Path(path_s).name

    # find unique mesh. ignore pose variants
    unique_names = set()
    for path in sil_f_paths:
        org_name = remove_pose_variant_in_file_name(path.stem)
        unique_names.add(org_name)

    unique_names = [name for name in unique_names]
    N_uniq = len(unique_names)

    # split the subject unique names into train, valid, test sets
    #np.random.seed(100)
    label = np.zeros(N_uniq, dtype=np.uint8)
    for idx, name in enumerate(unique_names):
        label[idx] = 1 if '_male' in name else 0
    org_train_idxs, org_test_idxs = train_test_split(np.arange(N_uniq), test_size=0.1, stratify=label)  # big test size for reduced traning time

    label = np.zeros(len(org_train_idxs), dtype=np.uint8)
    for i in range(len(org_train_idxs)):
        label[i] = 1 if '_male' in unique_names[org_train_idxs[i]] else 0
    org_train_idxs, org_valid_idxs = train_test_split(org_train_idxs, test_size=0.10, stratify=label)

    # classify subject/human names into train, valid, test
    name_classes = {}
    for idx in org_train_idxs:
        name_classes[unique_names[idx]] = 'train'
    for idx in org_valid_idxs:
        name_classes[unique_names[idx]] = 'valid'
    for idx in org_test_idxs:
        name_classes[unique_names[idx]] = 'test'

    # now we classify all pose variant paths in to train, valid, test sets
    # based on the corresponding subject/human name
    train_idxs = []
    valid_idxs = []
    test_idxs = []
    for idx, path in enumerate(sil_f_paths):
        org_name = remove_pose_variant_in_file_name(path.stem)
        class_id = name_classes[org_name]
        if class_id == 'train':
            train_idxs.append(idx)
        elif class_id == 'valid':
            valid_idxs.append(idx)
        elif class_id == 'test':
            test_idxs.append(idx)
        else:
            assert 'opp. something wrong. unexpected name format'

    return np.array(train_idxs), np.array(valid_idxs), np.array(test_idxs)

def split_train_valid_test(sil_f_paths, sil_s_paths):
    assert len(sil_f_paths) == len(sil_s_paths)
    for path_f, path_s in zip(sil_f_paths, sil_s_paths):
        assert Path(path_f).name == Path(path_s).name

    n = len(sil_f_paths_dict)
    n_females = len([name for name in sil_f_paths if '_female' in name])
    n_males = len([name for name in sil_f_paths if '_male' in name])
    print(f'n females = {n_females}, n_males = {n_males}')

    np.random.seed(100)
    label = np.zeros(len(sil_f_paths), dtype=np.uint8)
    for idx, path in enumerate(sil_f_paths):
        label[idx] = 1 if '_male' in path.stem else 0
    train_idxs, test_idxs = train_test_split(np.arange(n), test_size=0.1, stratify=label)  # big test size for reduced traning time

    label = np.zeros(len(train_idxs), dtype=np.uint8)
    for i in range(len(train_idxs)):
        label[i] = 1 if '_male' in sil_f_paths[train_idxs[i]].stem else 0
    train_idxs, valid_idxs = train_test_split(train_idxs, test_size=0.10, stratify=label)

    return train_idxs, valid_idxs, test_idxs



if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-sil_f_ml_dir",  default=True, required=False)
    ap.add_argument("-sil_s_ml_dir",  default=True, required=False)
    ap.add_argument("-sil_f_fml_dir",  default=True, required=False)
    ap.add_argument("-sil_s_fml_dir",  default=True, required=False)

    ap.add_argument("-target_ml_dir",  default=True, required=False)
    ap.add_argument("-target_fml_dir",  default=True, required=False)

    ap.add_argument("-pca_ml_model_path",  default=True, required=False)
    ap.add_argument("-pca_fml_model_path",  default=True, required=False)

    ap.add_argument("-out_dir", default=False, required=False)
    ap.add_argument("-resize_size", type=str,  default='360x360', required=False)
    ap.add_argument("-post_process",  action='store_true')
    ap.add_argument("-c", "--color", action='store_true',help="color body part parsing or binary silhouette?")
    ap.add_argument("-p", "--n_pose_variant", type=int, default=0 ,help="are there pose variants in the input images?: name0_pose0, name0_pose1, name0_pose30,..")
    args = ap.parse_args()

    size = args.resize_size.split('x')
    size = (int(size[0]), int(size[1]))

    os.makedirs(args.out_dir, exist_ok=True)

   # path = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj/blender_images/realistic_projections/female/front/CSR0017A.png'
   # segment_body_part_sil_f(path)
   # exit()

    #save pca model
    model_female = joblib.load(filename=args.pca_fml_model_path)
    model_male = joblib.load(filename=args.pca_ml_model_path)
    pca_model = PcaModel(model_female=model_female, model_male=model_male)
    out_path = os.path.join(*[args.out_dir, 'pca_model.jlb'])
    pca_model.dump(out_path)

    print(f'dump pca model to {out_path}')

    #n_file = 30*10 #for debugging with small number of files
    n_file = -1
    # copy pca target with the same name pattern to the out dir
    out_target_dir = os.path.join(*[args.out_dir, 'target'])
    #copy_target_prefix(args.target_fml_dir, out_target_dir, '_female', args.n_pose_variant, n_files=n_file)
    #copy_target_prefix(args.target_ml_dir,  out_target_dir, '_male', args.n_pose_variant, n_files=n_file)

    #out_height_path = os.path.join(*[args.out_dir, 'height.txt'])
    #dump_heights(pca_in_dir=out_target_dir,
    #             pca_ml_model_path=args.pca_ml_model_path, pca_fml_model_path=args.pca_fml_model_path,
    #             height_out_path=out_height_path)

    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f'created temporary dir: {tmp_dir}')

        tmp_sil_f_dir = os.path.join(*[tmp_dir, 'sil_f'])
        tmp_sil_s_dir = os.path.join(*[tmp_dir, 'sil_s'])

        # merge both male and female to a tempt dir and make their file names distinctive
        copy_file_prefix(args.sil_f_ml_dir, tmp_sil_f_dir, '_male', n_file=n_file)
        copy_file_prefix(args.sil_s_ml_dir, tmp_sil_s_dir, '_male',n_file=n_file)

        copy_file_prefix(args.sil_f_fml_dir, tmp_sil_f_dir, '_female',n_file=n_file)
        copy_file_prefix(args.sil_s_fml_dir, tmp_sil_s_dir, '_female',n_file=n_file)

        #remove_missing_pair(tmp_sil_f_dir, tmp_sil_s_dir)
        #make sure that there is a complete pari: front-side for every images
        verify_missing_pair(tmp_sil_f_dir, tmp_sil_s_dir)

        sil_f_paths_dict = dict([(path.stem, path) for path in Path(tmp_sil_f_dir).glob('*.*')])
        sil_s_paths_dict = dict([(path.stem, path) for path in Path(tmp_sil_s_dir).glob('*.*')])
        assert sil_f_paths_dict.keys() == sil_s_paths_dict.keys()

        sil_f_paths = [path for path in sil_f_paths_dict.values()]
        sil_s_paths = [path for path in sil_s_paths_dict.values()]

        name_ids = [id for id in sil_s_paths_dict.keys()]

        out_sil_f_dir = join(*[args.out_dir, 'sil_f'])
        out_sil_s_dir = join(*[args.out_dir, 'sil_s'])
        os.makedirs(out_sil_f_dir, exist_ok=True)
        os.makedirs(out_sil_s_dir, exist_ok=True)

        if args.n_pose_variant == 0:
            train_idxs, valid_idxs, test_idxs = split_train_valid_test(sil_f_paths, sil_s_paths)
        else:
            #if there are N_pose_variants per subject, we need to split in a way that all pose variant images of a subject
            #stay completely inside a set. There must be no cases like: subject0_pose_0 is in the train set but subject0_pose_25 is in the test set
            train_idxs, valid_idxs, test_idxs = split_train_valid_test_pose_variants(sil_f_paths, sil_s_paths)

            #verify if our splitting is correct
            verify_splitting_pose_variant(sil_f_paths, train_idxs=train_idxs, valid_idxs=valid_idxs, test_idxs=test_idxs, N_pose_per_subject=args.n_pose_variant)
            verify_splitting_pose_variant(sil_s_paths, train_idxs=train_idxs, valid_idxs=valid_idxs, test_idxs=test_idxs, N_pose_per_subject=args.n_pose_variant)

        copy_train_valid_test(name_ids, sil_f_paths_dict, train_idxs=train_idxs, valid_idxs=valid_idxs, test_idxs=test_idxs, out_dir=out_sil_f_dir, args=args)
        copy_train_valid_test(name_ids, sil_s_paths_dict, train_idxs=train_idxs, valid_idxs=valid_idxs, test_idxs=test_idxs, out_dir=out_sil_s_dir, args=args)

    print(f'deleted temporary dir')

    if not args.color:
        crop_train_test_valid(out_sil_f_dir, out_sil_s_dir, size)

