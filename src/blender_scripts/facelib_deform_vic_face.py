import bpy
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

resolution = 256
def load_fcl_keypoint_idxs(fcl_kp_path):
    uv_kpt_ind = np.loadtxt(fcl_kp_path).astype(np.int32)
    idxs = []
    for i in range(uv_kpt_ind.shape[1]):
        idx = uv_kpt_ind[1,i] + uv_kpt_ind[0,i]*256
        idxs.append(idx)
    return idxs

def select_single_obj(obj):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select = True
    bpy.context.scene.objects.active = obj


def main():
    fcl_kp_path = '/home/khanhhh/data_1/projects/Oh/data/face/meta_data/facelib_kp_mesh_idx.pkl'
    with open(fcl_kp_path, 'rb') as file:
        fcl_kp_idxs = pickle.load(file)
    print(fcl_kp_idxs)

    vic_kp_path = '/home/khanhhh/data_1/projects/Oh/codes/human_estimation/data/meta_data/face_keypoint.pkl'
    with open(vic_kp_path, 'rb') as file:
        data = pickle.load(file)
        vic_kp_idxs = data['keypoint_idx']

    fcl_obj = bpy.data.objects['facelib_obj']

    vic_mesh = bpy.data.objects['VictoriaMesh'].data
    vic_kp_coords = []
    for i in vic_kp_idxs:
        vic_kp_co = vic_mesh.vertices[i].co
        vic_kp_coords.append((vic_kp_co))
    
    fcl_bm = bmesh.from_edit_mesh(fcl_obj.data)
    for i, idx in enumerate(fcl_kp_idxs):
        fcl_bm.verts[idx].select = True
        for j in range(3):
            fcl_bm.verts[idx].co[j] = vic_kp_coords[i][j]
        #bpy.ops.object.hook_add_newob()

    #bpy.ops.object.mode_set(mode='OBJECT')
    #select_single_obj(fcl_obj)
    #modifier = fcl_obj.modifiers.new(name="Laplacian", type='LAPLACIANDEFORM')
    #modifier.vertex_group = 'kp'
    #bpy.ops.object.laplaciandeform_bind()

    

if __name__ == '__main__':
    main()