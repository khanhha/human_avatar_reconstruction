from pathlib import Path
import numpy as np
import os
import bpy
import bmesh

def scale_obj(obj, s = 0.01):
    select_single_obj(obj)
    #mesh = obj.data
    bpy.ops.transform.resize(value=(s,s,s))
    bpy.ops.object.transform_apply(location=False, scale=True, rotation=False)

def translate_obj(obj, t):
    select_single_obj(obj)
    bpy.ops.transform.translate(value=t)
    bpy.ops.object.transform_apply(location=False, scale=True, rotation=False)
    
def import_obj(path, name):
    bpy.ops.import_scene.obj(filepath=path, axis_forward='Y', axis_up='Z', split_mode='OFF')
    obj = bpy.context.selected_objects[0]
    obj.name = name
    return obj

def select_single_obj(obj):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select = True
    bpy.context.scene.objects.active = obj
    
dir_0 = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj/cnn_data/result/model_joint_pca_50/train/'
dir_1 = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj/caesar_obj/'

paths_0 = [path for path in Path(dir_0).glob('*.obj')]
paths_1 = dict([(path.name, path) for path in Path(dir_1).glob('*.obj')])

assert len(paths_0) > 0, dir_0+': empty directory '
assert len(paths_1) > 0, dir_1+': empty directory'

n = len(paths_0)
n_file = 25

square_range = 50
row_size = int(np.sqrt(n_file)) +1
x = np.linspace(-square_range, square_range, row_size)
y = np.linspace(-square_range, square_range, row_size)
loc_x, loc_y = np.meshgrid(x, y)
print('\n\n\n\nkhanh')
print(loc_x)
print(loc_y)


for i in range(n_file):
    idx = np.random.randint(0, n)
    path_0 = paths_0[idx]
    if path_0.name not in paths_1:
        print('missing file: ', path_0.name)
        continue

    path_1 = paths_1[path_0.name]
    obj_0 = import_obj(str(path_0), path_0.stem)
    obj_1 = import_obj(str(path_1), path_1.stem+'_groudtruth')
    select_single_obj(obj_1)
    scale_obj(obj_1)
    t = (5.0, 0.0, 0.0)
    translate_obj(obj_1, t)
    
    x_idx = i //  row_size
    y_idx = i % row_size
    t = (loc_x[x_idx, y_idx], 0.0, loc_y[x_idx, y_idx])
    print('loc = ', x_idx, y_idx, t)
    translate_obj(obj_0, t)
    translate_obj(obj_1, t)
    #print(path_0, path_1)





