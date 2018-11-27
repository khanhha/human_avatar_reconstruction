import pickle
from pathlib import Path
import shutil
import os

def split_male_females():
    path = '/home/khanhhh/data_1/projects/Oh/data/3d_human/debug/caesar_obj_gender.pkl'

    OUT_DIR = '/home/khanhhh/data_1/projects/Oh/data/3d_human/'
    OUT_MALE_DIR = f'{OUT_DIR}/caesar_obj_male/'
    OUT_FEMALE_DIR = f'{OUT_DIR}/caesar_obj_female/'
    os.makedirs(OUT_MALE_DIR, exist_ok=True)
    os.makedirs(OUT_FEMALE_DIR, exist_ok=True)

    IN_OBJ_DIR = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj/'

    males = set()
    females = set()
    with open(path, 'rb') as file:
        genders = pickle.load(file)
        for name, gender in genders.items():
            if gender == True:
                females.add(name)
            else:
                males.add(name)

    for path in Path(IN_OBJ_DIR).glob('*.obj'):
        if '_ld' in path.stem:
            continue
        if path.stem in males:
            dest_path = f'{OUT_MALE_DIR}/{path.name}'
        else:
            dest_path = f'{OUT_FEMALE_DIR}/{path.name}'

        shutil.copy(str(path), dest_path)

    print('females: ', len(females))
    print('males: ', len(males))

if __name__ == '__main__':
    split_male_females()