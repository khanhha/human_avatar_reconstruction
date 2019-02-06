from pathlib import Path
import shutil

if __name__ == '__main__':
    ALL_OBJ_DIR = '/media/khanhhh/42855ff5-574e-4a41-ad10-0f08087b0ff6/data_1/projects/Oh/data/3d_human/caesar_norm_wsx_obj/'
    FEMALE_DIR = '/media/khanhhh/42855ff5-574e-4a41-ad10-0f08087b0ff6/data_1/projects/Oh/data/3d_human/caesar_obj/female_meshes/'
    MALE_DIR = '/media/khanhhh/42855ff5-574e-4a41-ad10-0f08087b0ff6/data_1/projects/Oh/data/3d_human/caesar_obj/male_meshes/'

    NAME_DIR = '/media/khanhhh/42855ff5-574e-4a41-ad10-0f08087b0ff6/data_1/projects/Oh/data/3d_human/caesar_obj/slice_code/fourier/Aux_Hip_Waist_0'

    female_names = set([path.stem for path in Path(NAME_DIR).glob('*.pkl')])
    obj_paths = [path for path in Path(ALL_OBJ_DIR).glob('*.obj') if '_ld' not in path.stem]
    male_paths = []
    female_paths = []
    for path in obj_paths:
        if path.stem in female_names:
            female_paths.append(path)
        else:
            male_paths.append(path)

    for path in female_paths:
        shutil.copy(src=str(path), dst=f'{FEMALE_DIR}/{path.name}')
    for path in male_paths:
        shutil.copy(src=str(path), dst=f'{MALE_DIR}/{path.name}')


