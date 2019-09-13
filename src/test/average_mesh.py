import numpy  as np
from common.obj_util import import_mesh_obj, export_mesh
from pathlib import Path
from tqdm import tqdm
if __name__ == '__main__':
    #dir = '/media/khanhhh/42855ff5-574e-4a41-ad10-0f08087b0ff6/data_1/projects/Oh/data/3d_human/SPRING_MALE/'
    #out = '/home/khanhhh/data_1/projects/Oh/codes/human_estimation/data/meta_data_shared/mean_male_ucsc.obj'
    dir = '/media/khanhhh/42855ff5-574e-4a41-ad10-0f08087b0ff6/data_1/projects/Oh/data/3d_human/SPRING_FEMALE/'
    out = '/home/khanhhh/data_1/projects/Oh/codes/human_estimation/data/meta_data_shared/mean_female_ucsc.obj'
    paths = [path for path in Path(dir).glob('*.obj')]

    V = []
    verts, faces = import_mesh_obj(paths[0])
    for path in tqdm(paths):
        if 'SPRING' not in str(path):
            continue
        verts, _ = import_mesh_obj(path)
        V.append(verts.flatten())

    print(f'N obj files = {len(V)}')
    V = np.array(V)
    V = np.average(V, axis=0)
    V = V.reshape(V.shape[0]//3, 3)
    export_mesh(out, verts=V, faces=faces)