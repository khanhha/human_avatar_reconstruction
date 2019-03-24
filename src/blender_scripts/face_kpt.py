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


def find_mirror_vertices(obj, my_verts_idxs, error_threshold=.1):
    mesh = obj.data
    my_verts = []
    for idx in my_verts_idxs:
        my_verts.append(mesh.vertices[idx])

    assert len(my_verts) > 0

    size = len(mesh.vertices)
    kd = mathutils.kdtree.KDTree(size)
    for i,v in enumerate(mesh.vertices):
        kd.insert(v.co, i)
    kd.balance()

    mirror_idxs = []
    for idx, mv in enumerate(my_verts):
        mirror_co = deepcopy(mv.co)
        mirror_co.x = -mirror_co.x
        #dsts = np.array([(mirror_co - ov.co).length for ov in mesh.vertices])
        #idx = np.argmin(dsts)
        #found_co = mesh.vertices[idx].co
        #error = (found_co - mirror_co).length
        found_co, index, dst = kd.find(mirror_co)
        if dst > error_threshold:
            grp_names = [obj.vertex_groups[grp.group].name for grp in mv.groups]
            print('failed symmetric vertex for vertex ', mv, ' in the groups ', grp_names, 'use the vertex index -1 itself')
            #assert dst < error_threshold, 'distance to found symmetric vertex is large'
            index = -1
        mirror_idxs.append(index)

    assert len(my_verts_idxs) == len(mirror_idxs), 'not find enough mirrored points'
    print('found all mirrored vertices: ', len(mirror_idxs))

    return mirror_idxs


kpt  = 68*[-1]
kpt[0] = 39953
kpt[1] = 41472
kpt[2] = 41078
kpt[3] = 41097
kpt[4] = 41090
kpt[5] = 41171
kpt[6] = 41182
kpt[7] = 42626
kpt[8] = 32210

kpt[17] = 40979
kpt[18] = 40998
kpt[19] = 41507
kpt[20] = 41503
kpt[21] = 46140

kpt[27] = 38317
kpt[28] = 38337
kpt[29] = 38353
kpt[30] = 38387

kpt[31] = 40336
kpt[32] = 46506
kpt[33] = 38436

kpt[36] = 45862
kpt[37] = 45658
kpt[38] = 45500
kpt[39] = 45175
kpt[40] = 44941
kpt[41] = 44837

kpt[48] = 42553
kpt[49] = 42581
kpt[50] = 42597 
kpt[51] = 34656

kpt[57] = 34394
kpt[58] = 42366
kpt[58] = 42526
kpt[59] = 42365
kpt[60] = 49349
kpt[61] = 49387
kpt[62] = 48209

kpt[66] = 48443
kpt[67] = 50372

pairs  = [(9,7), (10, 6), (11,5), (12,4), (13,3), (14,2), (15,1), (16,0)]
pairs += [(22,21), (23,20), (24,19), (25,18), (26,17)]
pairs += [(42,39), (43,38), (44,37), (45,36), (46,41), (47,40)]
pairs += [(35,31), (34,32)]
pairs += [(52,50), (53,49), (54,48), (55,59), (56,58)]
pairs += [(64,60), (63,61), (65,67)]

obj = bpy.data.objects['VictoriaMesh']
bm = bmesh.from_edit_mesh(obj.data)

v_idxs = [kpt[pair[1]] for pair in pairs]
for idx in v_idxs:
    assert idx != -1

mv_idxs = find_mirror_vertices(obj, v_idxs)
print(mv_idxs)
for i, pair in enumerate(pairs):
    kpt[pair[0]] = mv_idxs[i]

kp_coords = []
for idx in kpt:
    assert idx != -1
    bm.verts[idx].select = True
    kp_coords.append(bm.verts[idx].co)
    

out_path = '/home/khanhhh/data_1/projects/Oh/codes/human_estimation/data/meta_data/face_keypoint.pkl'
with open(out_path, 'wb') as file:
    data = {'keypoint_idx':kpt, 'keypoint_co':kp_coords}
    pickle.dump(file=file, obj=data)
    

