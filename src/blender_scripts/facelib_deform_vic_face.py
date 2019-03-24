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
    fcl_bm = bmesh.from_edit_mesh(fcl_obj.data)

    vic_mesh = bpy.data.objects['VictoriaMesh'].data
    for i, idx in enumerate(fcl_kp_idxs):
        fcl_bm.verts[idx].select = True
        vic_kp_co = vic_mesh.vertices[vic_kp_idxs[i]].co
        cur_co = fcl_bm.verts[idx].co
        t = vic_kp_co - cur_co
        bpy.ops.transform.translate(value=t, proportional='ENABLED', proportional_size=0.3)
        if i >2:
            break
        break

if __name__ == '__main__':
    main()