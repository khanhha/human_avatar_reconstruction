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
ld_idxs = np.load(ld_path).flatten()


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


def export_mesh(fpath, verts, faces, add_one=True):
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
    faces = []
    with open(fpath, 'r') as obj:
        file = obj.read()
        lines = file.splitlines()
        for line in lines:
            elem = line.split()
            if elem:
                if elem[0] == 'v':
                    coords.append((float(elem[1]), float(elem[2]), float(elem[3])))
                elif elem[0] == 'vt' or elem[0] == 'vn' or elem[0] == 'vp':
                    # raise Exception('load obj file: un-supported texture, normal...')
                    continue
                elif elem[0] == 'f':
                    f = []
                    for v_idx in elem[1:]:
                        f.append(int(v_idx) - 1)
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
                    # space.show_outline_selected = True


def calc_caesar_mesh_height(DIR_IN_OBJ, PATH_OUT):
    ld_path = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_meta/landmarksIdxs73.npy'
    ld_idxs = np.load(ld_path)

    error_obj_paths = []

    heights = {}
    for i, path in enumerate(Path(DIR_IN_OBJ).glob('*.obj')):
        print(i, str(path))

        print(i, str(path))
        obj_caesar = import_obj(str(path), path.stem)
        transform_obj_caesar(obj_caesar, ld_idxs, s=0.1)

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


def project_silhouette_camera_configurations():
    DIR_SIL_F = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj/caesar_silhouette_diff_cam/sil_f_fc/'
    DIR_SIL_S = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj/caesar_silhouette_diff_cam/sil_s_fc/'

    os.makedirs(DIR_SIL_F, exist_ok=True)
    os.makedirs(DIR_SIL_S, exist_ok=True)
    for path in Path(DIR_SIL_F).glob('*.*'):
        os.remove(str(path))
    for path in Path(DIR_SIL_S).glob('*.*'):
        os.remove(str(path))


    obj_path = Path('/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj/caesar_obj/csr4003a.obj')
    obj_name = obj_path.stem
    if obj_name not in bpy.data.objects:
        obj_caesar = import_obj(str(obj_path), obj_name)
        transform_obj_caesar(obj_caesar, ld_idxs, s=0.1)
    else:
        obj_caesar = bpy.data.objects[obj_name]

    # cam_obj = bpy.data.objects['Camera']
    #
    # org_loc = cam_obj.location[:]
    # cam_y_range = range(-300, -700, -10)
    # for cam_y in cam_y_range:
    #     cam_obj.location[1] = cam_y
    #
    #     set_silhouette_silhouette_mode()
    #     front_sil_path = os.path.join(*[DIR_SIL_F, obj_path.stem + '_dst_' + str(abs(cam_y))])
    #     bpy.data.scenes['Scene'].render.filepath = front_sil_path
    #     bpy.ops.render.opengl(write_still=True, view_context=True)
    #
    #     # side veiw
    #     select_single_obj(obj_caesar)
    #     bpy.ops.transform.rotate(value=-np.pi / 2.0, axis=(0.0, 0.0, 1.0))
    #     bpy.ops.object.transform_apply(location=True, scale=True, rotation=True)
    #
    #     side_sil_path = os.path.join(*[DIR_SIL_S, obj_path.stem + '_dst_' + str(abs(cam_y))])
    #     bpy.data.scenes['Scene'].render.filepath = side_sil_path
    #     bpy.ops.render.opengl(write_still=True, view_context=True)
    #
    #     # back to front view
    #     select_single_obj(obj_caesar)
    #     bpy.ops.transform.rotate(value=np.pi / 2.0, axis=(0.0, 0.0, 1.0))
    #     bpy.ops.object.transform_apply(location=True, scale=True, rotation=True)
    #
    # cam_obj.location[:] = org_loc

    # cam_obj = bpy.data.objects['Camera']
    #
    # org_loc = cam_obj.location[:]
    # cam_z_range = range(70, 150, 10)
    # for idx, cam_z in enumerate(cam_z_range):
    #     cam_obj.location[2] = cam_z
    #
    #     set_silhouette_silhouette_mode()
    #     front_sil_path = os.path.join(*[DIR_SIL_F, obj_path.stem + '_' + str(idx) + '_z_' + str(abs(cam_z))])
    #     bpy.data.scenes['Scene'].render.filepath = front_sil_path
    #     bpy.ops.render.opengl(write_still=True, view_context=True)
    #
    #     # side veiw
    #     select_single_obj(obj_caesar)
    #     bpy.ops.transform.rotate(value=-np.pi / 2.0, axis=(0.0, 0.0, 1.0))
    #     bpy.ops.object.transform_apply(location=True, scale=True, rotation=True)
    #
    #     side_sil_path = os.path.join(*[DIR_SIL_S, obj_path.stem + '_' + str(idx) + '_z_' + str(abs(cam_z))])
    #     bpy.data.scenes['Scene'].render.filepath = side_sil_path
    #     bpy.ops.render.opengl(write_still=True, view_context=True)
    #
    #     # back to front view
    #     select_single_obj(obj_caesar)
    #     bpy.ops.transform.rotate(value=np.pi / 2.0, axis=(0.0, 0.0, 1.0))
    #     bpy.ops.object.transform_apply(location=True, scale=True, rotation=True)
    #
    # cam_obj.location[:] = org_loc

    cam_obj = bpy.data.cameras['Camera']
    org_len = cam_obj.lens
    step = 0.15
    focal_lens = np.arange(2.5, 5.0+step, step)
    for focal_l in focal_lens:
        cam_obj.lens = focal_l

        set_silhouette_silhouette_mode()
        front_sil_path = os.path.join(*[DIR_SIL_F, obj_path.stem + '_focal_len_' + str(abs(focal_l))])
        bpy.data.scenes['Scene'].render.filepath = front_sil_path
        bpy.ops.render.opengl(write_still=True, view_context=True)

        # side veiw
        select_single_obj(obj_caesar)
        bpy.ops.transform.rotate(value=-np.pi / 2.0, axis=(0.0, 0.0, 1.0))
        bpy.ops.object.transform_apply(location=True, scale=True, rotation=True)

        side_sil_path = os.path.join(*[DIR_SIL_S, obj_path.stem + '_focal_len_' + str(abs(focal_l))])
        bpy.data.scenes['Scene'].render.filepath = side_sil_path
        bpy.ops.render.opengl(write_still=True, view_context=True)

        # back to front view
        select_single_obj(obj_caesar)
        bpy.ops.transform.rotate(value=np.pi / 2.0, axis=(0.0, 0.0, 1.0))
        bpy.ops.object.transform_apply(location=True, scale=True, rotation=True)

    cam_obj.lens = org_len

project_silhouette_camera_configurations()