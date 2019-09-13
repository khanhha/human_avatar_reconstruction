from pathlib import Path
import os
import cv2 as cv
from pca.nn_util import remove_pose_variant_in_file_name, verify_pose_variants_per_name
from tqdm import tqdm
import shutil
if __name__ == '__main__':
    dir0 = '/media/khanhhh/42855ff5-574e-4a41-ad10-0f08087b0ff6/data_1/projects/Oh/data/3d_human/caesar_obj/blender_images/nosyn/male/sil_s_raw/'
    dir1 = '/media/khanhhh/42855ff5-574e-4a41-ad10-0f08087b0ff6/data_1/projects/Oh/data/3d_human/caesar_obj/blender_images/nosyn/male/sil_f_raw/'
    dir2 = '/media/khanhhh/42855ff5-574e-4a41-ad10-0f08087b0ff6/data_1/projects/Oh/data/3d_human/caesar_obj/blender_images/nosyn/female/sil_s_raw/'
    dir3 = '/media/khanhhh/42855ff5-574e-4a41-ad10-0f08087b0ff6/data_1/projects/Oh/data/3d_human/caesar_obj/blender_images/nosyn/female/sil_f_raw/'

    error_files = []
    dirs = [dir0, dir1, dir2, dir3]
    for dir in dirs:
        print(f'process dir: {dir}')
        paths = [path for path in Path(dir).glob('*.*')]
        for path in tqdm(paths):
            img = cv.imread(str(path))
            if img is None:
                error_files.append(error_files)
                print(f'error file: {str(path)}')
                os.remove(str(path))
            del img

   # for dir in dirs:
   #     paths = [path for path in Path(dir).glob('*.*')]
   #     paths = sorted(paths)
   #     N = len(paths)
   #     n_pose_variants = 30
   #     verify_pose_variants_per_name(paths, n_pose_variants)
    exit()
