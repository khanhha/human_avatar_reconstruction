from pathlib import Path
import bpy
from os.path import join
import os
import pickle
import numpy as  np
import mathutils.geometry as geo
from mathutils import Vector
from bpy import context
import bmesh
from collections import defaultdict
from copy import deepcopy
import mathutils


def select_single_obj(obj):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select = True
    bpy.context.scene.objects.active = obj

def deselect_all_obj():
    bpy.ops.object.select_all(action='DESELECT')

def load_obj(path, obj_name):
    verts = []
    with open(path, 'r') as file:
        lines = file.readlines()
        for l in lines:
            if l[0] != 'v':
                continue
            x,y,z = l.split(' ')[1:4]
            x,y,z = float(x), float(y), float(z)
            verts.append((x,y,z))

    obj = create_points_obj(verts, obj_name)
    return obj

def create_points_obj(verts, name):
    # Define mesh and object
    mesh = bpy.data.meshes.new("mesh")
    # the mesh variable is then referenced by the object variable
    obj = bpy.data.objects.new(name, mesh)

    bpy.context.scene.objects.link(obj) # linking the object to the scene
    mesh.from_pydata(verts, [], [])
    mesh.update(calc_edges=False)
    return obj

def load_keypoints(path, obj_name):
    verts = []
    with open(path, 'r') as file:
        lines = file.readlines()
        for l in lines:
            x,y,z = l.split(' ')
            x,y,z = float(x), float(y), float(z)
            verts.append((x,y,z))

    obj = create_points_obj(verts, obj_name)
    return obj

def find_kp_face_mesh_indices(kp_obj, fc_obj):
    fmesh = fc_obj.data

    size = len(fmesh.vertices)
    kd = mathutils.kdtree.KDTree(size)
    for i, v in enumerate(fmesh.vertices):
        kd.insert(v.co, i)
    kd.balance()

    kp_idxs = []
    for idx, mv in enumerate(kp_obj.data.vertices):
        found_co, index, dst = kd.find(mv.co)
        kp_idxs.append(index)

    return kp_idxs


def shift_objects_to_org(obj, kp_obj):
    verts = [v.co[:] for v in kp_obj.data.vertices]
    center = np.mean(verts, axis=0)

    for v in obj.data.vertices:
        for j in range(3):
            v.co[j] -= center[j]

    for v in kp_obj.data.vertices:
        for j in range(3):
            v.co[j] -= center[j]

def main():
    in_dir = '/media/khanhhh/42855ff5-574e-4a41-ad10-0f08087b0ff6/data_1/projects/Oh/data/face/prn_output/'
    names = [path.stem for path in Path(in_dir).glob('*.obj')]

    out_kp_idx_path = '/home/khanhhh/data_1/projects/Oh/data/face/meta_data/facelib_kp_mesh_idx.pkl'
    for name in names:
        if 'front_IMG_1928' not in name:
            continue

        obj_path = join(*[in_dir, name+'.obj'])
        deselect_all_obj()
        bpy.ops.import_scene.obj(filepath=obj_path, axis_forward='Y', axis_up = 'Z')
        obj = bpy.context.selected_objects[0]
        obj.name = name
        #obj = load_obj(obj_path, name)

        #select_single_obj(obj)
        #bpy.ops.object.origin_set(type='GEOMETRY_ORIGIN', center='MEDIAN')

        kp_obj_name = name+'_kpt'
        kp_path = join(*[in_dir, kp_obj_name + '.txt'])
        obj_kp = load_keypoints(kp_path, kp_obj_name)

        kp_mesh_idxs = find_kp_face_mesh_indices(obj_kp, obj)
        with open(out_kp_idx_path, 'wb') as file:
            pickle.dump(file=file, obj=kp_mesh_idxs)

        #shift_objects_to_org(obj, obj_kp)

if __name__ == '__main__':
    main()

