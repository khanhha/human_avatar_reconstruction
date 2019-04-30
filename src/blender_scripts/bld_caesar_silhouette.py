import bpy
from mathutils import Vector
from pathlib import Path
import pickle as pkl
import numpy as np
import math
import os
from collections import defaultdict
import shutil
import copy
import scipy
from sklearn.externals import joblib
from sklearn.decomposition import IncrementalPCA
import pickle
import sys

scene = bpy.context.scene
g_cur_file_name = ''
ld_path = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_meta/landmarksIdxs73.npy'
ld_idxs = np.load(ld_path)

def import_obj(path, name):
    bpy.ops.import_scene.obj(filepath=path, axis_forward='Y', axis_up='Z', split_mode='OFF')
    obj = bpy.context.selected_objects[0]
    obj.name = name
    return obj


def select_single_obj(obj):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select = True
    bpy.context.scene.objects.active = obj


def copy_obj(obj, name, location):
    select_single_obj(obj)
    bpy.ops.object.duplicate()
    obj = scene.objects.active
    obj.name = name
    return obj


def delete_obj(obj):
    select_single_obj(obj)
    bpy.ops.object.delete()


def export_mesh(fpath, verts, faces, add_one = True):
    with open(fpath, 'w') as f:
        for i in range(verts.shape[0]):
            co = tuple(verts[i, :])
            f.write("v %.8f %.8f %.8f \n" % co)

        for i in range(len(faces)):
            f.write("f")
            for v_idx in faces[i]:
                if add_one == True:
                    v_idx += 1
                f.write(" %d" % (v_idx))
            f.write("\n")

def import_mesh(fpath):
    coords = []
    faces =  []
    with open(fpath, 'r') as obj:
        file = obj.read()
        lines = file.splitlines()
        for line in lines:
            elem = line.split()
            if elem:
                if elem[0] == 'v':
                    coords.append((float(elem[1]), float(elem[2]), float(elem[3])))
                elif elem[0] == 'vt' or elem[0] == 'vn' or elem[0] == 'vp':
                    #raise Exception('load obj file: un-supported texture, normal...')
                    continue
                elif elem[0] == 'f':
                    f = []
                    for v_idx in elem[1:]:
                        f.append(int(v_idx)-1)
                    faces.append(f)

    return np.array(coords), faces

def transform_obj_caesar(obj, ld_idxs, s=0.01):
    mesh = obj.data

    bpy.ops.transform.resize(value=(s, s, s))
    bpy.ops.object.transform_apply(location=True, scale=True, rotation=True)

    org = mesh.vertices[ld_idxs[72]].co
    bpy.ops.transform.translate(value=-org)
    bpy.ops.object.transform_apply(location=True, scale=False, rotation=False)

    p0_x = mesh.vertices[ld_idxs[16]].co
    p1_x = mesh.vertices[ld_idxs[18]].co
    x = p1_x - p0_x
    x.normalize()
    angle = x.dot(Vector((1.0, 0.0, 0.0)))
    # angle =  math.degrees(math.acos(angle))
    angle = math.acos(angle)
    print('angle', angle)
    bpy.ops.transform.rotate(value=angle, axis=(0.0, 0.0, 1.0))

    select_single_obj(obj)
    bpy.ops.object.transform_apply(location=True, scale=True, rotation=True)

def transform_obj_caesar_pca(obj, s=0.01):

    bpy.ops.transform.resize(value=(s, s, s))
    bpy.ops.object.transform_apply(location=True, scale=True, rotation=True)

    mesh = obj.data
    org = Vector(mesh.vertices[7652].co[:])
    org[2] = 0.0
    org[0] = 0.0
    bpy.ops.transform.translate(value=-org)
    bpy.ops.object.transform_apply(location=True, scale=False, rotation=False)

    select_single_obj(obj)
    bpy.ops.object.transform_apply(location=True, scale=True, rotation=True)


def collect_vertex_group_idxs(obj, name):
    idxs = []
    for idx, v in enumerate(obj.data.vertices):
        for vgrp in v.groups:
            grp_name = obj.vertex_groups[vgrp.group].name
            if grp_name == name:
                idxs.append(idx)
    return idxs

def mpii_is_correct_mesh(obj):
    n_verts = len(obj.data.vertices)
    n_faces = len(obj.data.polygons)
    if n_verts != 6449 or n_faces != 12894:
        return False
    else:
        return True

def set_realistic_img_view_mode():
    for area in bpy.context.screen.areas:  # iterate through areas in current screen
        if area.type == 'VIEW_3D':
            for space in area.spaces:  # iterate through spaces in current VIEW_3D area
                if space.type == 'VIEW_3D':  # check if space is a 3D view
                    space.viewport_shade = 'SOLID'  # set the viewport shading to rendered
                    space.use_matcap = True
                    space.show_outline_selected = False


def set_silhouette_silhouette_mode():
    for area in bpy.context.screen.areas:  # iterate through areas in current screen
        if area.type == 'VIEW_3D':
            for space in area.spaces:  # iterate through spaces in current VIEW_3D area
                if space.type == 'VIEW_3D':  # check if space is a 3D view
                    space.viewport_shade = 'TEXTURED'  # set the viewport shading to rendered
                    #space.show_outline_selected = True

def calc_caesar_mesh_height(DIR_IN_OBJ, PATH_OUT):
    ld_path = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_meta/landmarksIdxs73.npy'
    ld_idxs = np.load(ld_path)

    error_obj_paths = []

    heights = {}
    for i, path in enumerate(Path(DIR_IN_OBJ).glob('*.obj')):
        print(i, str(path))

        print(i, str(path))
        obj_caesar = import_obj(str(path), path.stem)
        transform_obj_caesar(obj_caesar, ld_idxs)

        if not mpii_is_correct_mesh(obj_caesar):
            print('error object: ', path.stem)
            error_obj_paths.append(str(path))
            delete_obj(obj_caesar)
            continue

        min_z = np.finfo(float).max
        max_z = np.finfo(float).min
        mesh = obj_caesar.data
        for v in mesh.vertices:
            min_z = min(v.co.z, min_z)
            max_z = max(v.co.z, max_z)
        h = max_z - min_z
        heights[path.stem] = h

        delete_obj(obj_caesar)

    lines = []
    for name, h in heights.items():
        line = name + '_front.jpg ' + name + '_side.jpg ' + str(h) + '\n'
        lines.append(line)

    with open(PATH_OUT, 'a') as file:
        file.writelines(lines)

def project_silhouette_camera_configurations(DIR_IN_OBJ, obj_name, DIR_SIL_F, DIR_SIL_S):
    if '.obj' not in obj_name:
        obj_name = obj_name + '.obj'
    obj_path = Path(os.path.join(*[DIR_IN_OBJ, obj_name]))

    obj_caesar = import_obj(str(obj_path), obj_name)
    transform_obj_caesar(obj_caesar, ld_idxs)

    cam_obj = bpy.data.objects['Camera']

    if not mpii_is_correct_mesh(obj_caesar):
        print('error object: ', obj_name)
    else:
        org_loc = cam_obj.location[:]
        cam_y_range = range(-50,-30+5, 5)
        for cam_y in cam_y_range:
            cam_obj.location[1] = cam_y

            set_silhouette_silhouette_mode()
            front_sil_path = os.path.join(*[DIR_SIL_F, obj_path.stem + '_dst_'+ str(abs(cam_y))])
            bpy.data.scenes['Scene'].render.filepath = front_sil_path
            bpy.ops.render.opengl(write_still=True, view_context=True)

            # side veiw
            select_single_obj(obj_caesar)
            bpy.ops.transform.rotate(value=-np.pi / 2.0, axis=(0.0, 0.0, 1.0))
            bpy.ops.object.transform_apply(location=True, scale=True, rotation=True)

            side_sil_path = os.path.join(*[DIR_SIL_S, obj_path.stem + '_dst_'+ str(abs(cam_y))])
            bpy.data.scenes['Scene'].render.filepath = side_sil_path
            bpy.ops.render.opengl(write_still=True, view_context=True)

            #back to front view
            select_single_obj(obj_caesar)
            bpy.ops.transform.rotate(value=np.pi / 2.0, axis=(0.0, 0.0, 1.0))
            bpy.ops.object.transform_apply(location=True, scale=True, rotation=True)

        cam_obj.location[:] = org_loc

        cam = bpy.data.cameras['Camera']
        org_len = cam.lens
        step = 0.3
        focal_lens = np.arange(2.5, 4.10+0.3, step) +  0.1
        for focal_l in focal_lens:
            cam.lens = focal_l

            set_silhouette_silhouette_mode()
            front_sil_path = os.path.join(*[DIR_SIL_F, obj_path.stem + '_focal_len_'+ str(abs(focal_l))])
            bpy.data.scenes['Scene'].render.filepath = front_sil_path
            bpy.ops.render.opengl(write_still=True, view_context=True)

            # side veiw
            select_single_obj(obj_caesar)
            bpy.ops.transform.rotate(value=-np.pi / 2.0, axis=(0.0, 0.0, 1.0))
            bpy.ops.object.transform_apply(location=True, scale=True, rotation=True)

            side_sil_path = os.path.join(*[DIR_SIL_S, obj_path.stem + '_focal_len_'+ str(abs(focal_l))])
            bpy.data.scenes['Scene'].render.filepath = side_sil_path
            bpy.ops.render.opengl(write_still=True, view_context=True)

            #back to front view
            select_single_obj(obj_caesar)
            bpy.ops.transform.rotate(value=np.pi / 2.0, axis=(0.0, 0.0, 1.0))
            bpy.ops.object.transform_apply(location=True, scale=True, rotation=True)

        cam.lens = org_len

    #delete_obj(obj_caesar)


def project_front_side(DIR_IN_OBJ, DIR_SIL_F,  DIR_SIL_S, DIR_OUT_IMG = None, test_name = None):
    ld_idxs = np.load(ld_path)

    for path in Path(DIR_SIL_F).glob('*.*'):
        os.remove(str(path))
    for path in Path(DIR_SIL_S).glob('*.*'):
        os.remove(str(path))

    os.makedirs(DIR_SIL_F, exist_ok=True)
    os.makedirs(DIR_SIL_S, exist_ok=True)
    if DIR_OUT_IMG is not None:
        os.makedirs(DIR_OUT_IMG, exist_ok=True)

    error_obj_paths = []

    for i, path in enumerate(Path(DIR_IN_OBJ).glob('*.obj')):
        if test_name is not None and test_name not in path.stem:
            continue
        
        print(i, str(path))

        obj_caesar = import_obj(str(path), path.stem)
        transform_obj_caesar(obj_caesar, ld_idxs)

        if not mpii_is_correct_mesh(obj_caesar):
            print('error object: ', path.stem)
            error_obj_paths.append(str(path))
            delete_obj(obj_caesar)
            continue

        # front view
        if DIR_OUT_IMG is not None:
            set_realistic_img_view_mode()
            front_path = DIR_OUT_IMG + path.stem + '_front'
            bpy.data.scenes['Scene'].render.filepath = front_path
            bpy.ops.render.opengl(write_still=True, view_context=True)

        set_silhouette_silhouette_mode()
        front_sil_path = os.path.join(*[DIR_SIL_F, path.stem])
        bpy.data.scenes['Scene'].render.filepath = front_sil_path
        bpy.ops.render.opengl(write_still=True, view_context=True)

        # side veiw
        bpy.ops.transform.rotate(value=-np.pi / 2.0, axis=(0.0, 0.0, 1.0))
        bpy.ops.object.transform_apply(location=True, scale=True, rotation=True)

        if DIR_OUT_IMG is not None:
            set_realistic_img_view_mode()
            side_path = DIR_OUT_IMG + path.stem + '_side'
            bpy.data.scenes['Scene'].render.filepath = side_path
            bpy.ops.render.opengl(write_still=True, view_context=True)

            set_silhouette_silhouette_mode()

        side_sil_path = os.path.join(*[DIR_SIL_S, path.stem])
        bpy.data.scenes['Scene'].render.filepath = side_sil_path
        bpy.ops.render.opengl(write_still=True, view_context=True)
        
        if test_name is not None:
            return
        
        delete_obj(obj_caesar)
        # break

def project_synthesized():
    vic_mesh_path = '/home/khanhhh/data_1/projects/Oh/codes/human_estimation/data/meta_data/align_source_vic_mpii.obj'

    pca_path     = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj/vic_pca_models/pca_model_vic_male.jlb'
    pca_co_dir = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj/vic_pca_models/pca_coords/male/'
    sil_root_dir = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj/caesar_images_iphone_male/'

    sil_f_dir = os.path.join(*[sil_root_dir , 'sil_f_raw'])
    sil_s_dir = os.path.join(*[sil_root_dir , 'sil_s_raw'])
    os.makedirs(sil_f_dir, exist_ok=True)
    os.makedirs(sil_s_dir, exist_ok=True)
    for path in Path(sil_f_dir).glob('*.*'):
        os.remove(str(path))
    for path in Path(sil_s_dir).glob('*.*'):
        os.remove(str(path))

    pca_model = joblib.load(pca_path)

    _, tpl_faces = import_mesh(vic_mesh_path)

    vic_tpl_obj = bpy.data.objects['vic_template']
    vic_tpl_mesh = vic_tpl_obj.data
    n_v = len(vic_tpl_mesh.vertices)

    rarm_sample_v_idx = 5245 #a sample vertex on right arm to collapse arm
    larm_sample_v_idx = 3876 #a sample vertex on left arm to collapse arm
    larm_v_idxs = collect_vertex_group_idxs(vic_tpl_obj, 'larm')
    rarm_v_idxs = collect_vertex_group_idxs(vic_tpl_obj, 'rarm')

    paths = sorted([path for path in Path(pca_co_dir).glob('*.npy')])
    n = len(paths)

    #multiple blender instances for performance
    mid_idx = int(n/4)
    segments = [range(0, mid_idx), range(mid_idx, 2*mid_idx), range(2*mid_idx, 3*mid_idx), range(3*mid_idx, n)]

    blender_id = 0  #start blender instances with id in range [0,len(segments)]
    assert blender_id <= len(segments)

    for i in segments[blender_id]:
        if i >= n:
            continue

        path = paths[i]
        name = path.stem
        p = np.load(str(path))

        verts = pca_model.inverse_transform(p)
        verts = verts.reshape(n_v, 3)

        for vi in range(n_v):
            vic_tpl_mesh.vertices[vi].co[:] = verts[vi,:]

        transform_obj_caesar_pca(vic_tpl_obj, s = 10.0)

        set_silhouette_silhouette_mode()
        front_sil_path = os.path.join(*[sil_f_dir, name])
        bpy.data.scenes['Scene'].render.filepath = front_sil_path
        bpy.ops.render.opengl(write_still=True, view_context=True)

        # side veiw
        bpy.ops.transform.rotate(value=-np.pi / 2.0, axis=(0.0, 0.0, 1.0))
        bpy.ops.object.transform_apply(location=True, scale=True, rotation=True)

        #collapse left and right arm to avoid arm intrusion in the side profile
        larm_co = vic_tpl_mesh.vertices[larm_sample_v_idx].co[:]
        for vi in larm_v_idxs:
            vic_tpl_mesh.vertices[vi].co[:] = larm_co

        rarm_co = vic_tpl_mesh.vertices[rarm_sample_v_idx].co[:]
        for vi in rarm_v_idxs:
            vic_tpl_mesh.vertices[vi].co[:] = rarm_co

        side_sil_path = os.path.join(*[sil_s_dir, name])
        bpy.data.scenes['Scene'].render.filepath = side_sil_path
        bpy.ops.render.opengl(write_still=True, view_context=True)

project_synthesized()


# ROOT_DIR = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj/caesar_images_iphone/'
# DIR_IN_OBJ = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj/caesar_obj/'
# DIR_OUT_IMG = ROOT_DIR + 'caesar_front_side_images_raw/'
# DIR_SIL_F = os.path.join(*[ROOT_DIR , 'sil_f_raw'])
# DIR_SIL_S = os.path.join(*[ROOT_DIR , 'sil_s_raw'])
# test_name = None
# #test_name = 'CSR0309A'
# project_front_side(DIR_IN_OBJ, DIR_SIL_F=DIR_SIL_F, DIR_SIL_S=DIR_SIL_S, DIR_OUT_IMG = None, test_name=test_name)

#DIR_SIL_F = os.path.join(*[ROOT_DIR , test_name+'_sil_f'])
#DIR_SIL_S = os.path.join(*[ROOT_DIR , test_name+'_sil_s'])
#os.makedirs(DIR_SIL_F, exist_ok=True)
#os.makedirs(DIR_SIL_S, exist_ok=True)
#project_silhouette_camera_configurations(DIR_IN_OBJ, test_name,  DIR_SIL_F=DIR_SIL_F, DIR_SIL_S=DIR_SIL_S)

#height_path = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj/caesar_female_front_side_pair.txt'
#calc_caesar_mesh_height(DIR_IN_OBJ = DIR_IN_OBJ, PATH_OUT=height_path)
