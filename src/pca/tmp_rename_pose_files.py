from pathlib import Path
import os
if __name__ == '__main__':
    #dir = '/media/khanhhh/42855ff5-574e-4a41-ad10-0f08087b0ff6/data_1/projects/Oh/data/3d_human/caesar_obj/blender_images/nosyn/male/sil_s_raw/'
    dir = '/media/khanhhh/42855ff5-574e-4a41-ad10-0f08087b0ff6/data_1/projects/Oh/data/3d_human/caesar_obj/blender_images/nosyn/male/sil_f_raw/'
    #dir = '/media/khanhhh/42855ff5-574e-4a41-ad10-0f08087b0ff6/data_1/projects/Oh/data/3d_human/caesar_obj/blender_images/nosyn/female/sil_s_raw/'
    #dir = '/media/khanhhh/42855ff5-574e-4a41-ad10-0f08087b0ff6/data_1/projects/Oh/data/3d_human/caesar_obj/blender_images/nosyn/female/sil_f_raw/'
    for path in Path(dir).glob('*.*'):
        print(path)
        #idx = path.stem.rfind('_')+1
        #new_name = path.stem[:idx] + f'pose{path.stem[idx:]}' + path.suffix
        #new_path = os.path.join(*[path.parent, new_name])
        #os.rename(str(path), new_path)