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
import sys

scene = bpy.context.scene
g_cur_file_name = ''

def select_single_obj(obj):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select = True
    bpy.context.scene.objects.active = obj

def delete_obj(obj):
    select_single_obj(obj)
    bpy.ops.object.delete()

def transform_obj_caesar_pca(obj, s=0.01):

    select_single_obj(obj)

    bpy.ops.transform.resize(value=(s, s, s))
    bpy.ops.object.transform_apply(location=True, scale=True, rotation=True)

    mesh = obj.data
    org = Vector(mesh.vertices[7652].co[:]) #the crotch point
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

def find_heightest_vert_idx(obj, v_idxs):
    vert_cos = np.array([obj.data.vertices[idx].co[:] for idx in v_idxs])
    max_z_idx = int(np.argmax(vert_cos[:, 2]))
    return v_idxs[max_z_idx]

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

def avg_co(mesh, ld_idxs):
    avg_co = Vector((0.0, 0.0, 0.0))
    for idx in ld_idxs:
        avg_co += mesh.vertices[idx].co

    avg_co /= len(ld_idxs)
    return avg_co

def project_synthesized():
    pca_co_dir = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj/vic_pca_models_nosyn/verts/female/'
    sil_root_dir = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj/blender_images/realistic_projections/female/'
    n_files = 30
    
    sil_f_dir = os.path.join(*[sil_root_dir , 'front'])
    sil_s_dir = os.path.join(*[sil_root_dir , 'side'])
    os.makedirs(sil_f_dir, exist_ok=True)
    os.makedirs(sil_s_dir, exist_ok=True)
    
    from_scratch = True
    if from_scratch: 
        for path in Path(sil_f_dir).glob('*.*'):
            os.remove(str(path))
        for path in Path(sil_s_dir).glob('*.*'):
            os.remove(str(path))

    vic_tpl_obj = bpy.data.objects['vic_template']
    vic_tpl_mesh = vic_tpl_obj.data
    n_v = len(vic_tpl_mesh.vertices)
    neck_v_idxs = collect_vertex_group_idxs(vic_tpl_obj, 'neck_landmark')

    paths = sorted([path for path in Path(pca_co_dir).glob('*.npy')])

    cam_obj = bpy.data.objects['Camera']

    for i in range(n_files):
        path = paths[i]
        name = path.stem

        front_sil_path = os.path.join(*[sil_f_dir, name])
        side_sil_path = os.path.join(*[sil_s_dir, name])
        if Path(front_sil_path+'.png').exists() and  Path(side_sil_path+'.png').exists():
            continue
                
        verts = np.load(path)
        
        for vi in range(n_v):
            vic_tpl_mesh.vertices[vi].co[:] = verts[vi,:]

        transform_obj_caesar_pca(vic_tpl_obj, s = 10.0)

        neck_ld_co = avg_co(vic_tpl_mesh, neck_v_idxs)
        cam_obj.location[2] = neck_ld_co[2]

        #set_realistic_img_view_mode()

        bpy.data.scenes['Scene'].render.filepath = front_sil_path
        bpy.ops.render.opengl(write_still=True, view_context=True)
        
        # side veiw
        bpy.ops.transform.rotate(value=-np.pi / 2.0, axis=(0.0, 0.0, 1.0))
        bpy.ops.object.transform_apply(location=True, scale=True, rotation=True)
        
        #set_realistic_img_view_mode()
        bpy.data.scenes['Scene'].render.filepath = side_sil_path
        bpy.ops.render.opengl(write_still=True, view_context=True)
        
project_synthesized()
