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

def find_effective_cdd_triangles(tpl_obj, ctl_obj):
    tpl_mesh = tpl_obj.data
    ctl_mesh = ctl_obj.data
    print('start finding effective candidate control triangles for each vertex of Victoria')
    ctl_bm = bmesh.new()
    ctl_bm.from_mesh(ctl_mesh)

    ctl_bm.faces.ensure_lookup_table()
    ctl_bm.verts.ensure_lookup_table()

    bvh = mathutils.bvhtree.BVHTree.FromBMesh(ctl_bm)
    

    cdd_tris = []
    for idx, v in enumerate(tpl_mesh.vertices):
        
        #is_hand = False
        #for vgrp in v.groups:
        #    grp_name = tpl_obj.vertex_groups[vgrp.group].name
        #    if 'hand' in grp_name:
        #        is_hand = True
        #
            
        #if is_hand == True:
        ##    n_ring=1
        #else:
        #    n_ring=8
        
        n_ring = 8   
 

        ret = bvh.find_nearest(v.co)

        f_idx = ret[2]
        f = ctl_bm.faces[f_idx]

        cur_faces = {f.index}
        cur_verts = {v.index for v in f.verts}
        outer_verts = {v.index for v in f.verts}

        cnt = 0
        while cnt < n_ring:
            for v_idx in outer_verts:
                v = ctl_bm.verts[v_idx]
                for adj_f in v.link_faces:
                    cur_faces.add(adj_f.index)

            outer_verts = set()
            for f_idx in cur_faces:
                f = ctl_bm.faces[f_idx]
                for v in f.verts:
                    if v.index not in cur_verts:
                        outer_verts.add(v.index)
                        cur_verts.add(v.index)
            cnt += 1
        if idx == 23383:
            print('n_ring: ', n_ring, 'n_faces: ', len(cur_faces))
        #print('found ', len(cur_faces), 'faces for vertex ', idx)
        cdd_tris.append(list(cur_faces))
          
    return cdd_tris


tpl_obj = bpy.data.objects['source']
ctl_obj = bpy.data.objects['target']
cdd_tris = find_effective_cdd_triangles(tpl_obj, ctl_obj)
opath = "/home/khanhhh/data_1/projects/Oh/codes/human_estimation/data/meta_data/tpl_ctl_effective_cdd_tris_vic_mpii.pkl"
with open(opath, 'wb') as file:
    pickle.dump(cdd_tris, file)
