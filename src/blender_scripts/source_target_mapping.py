import bpy
import bmesh
import pickle
import mathutils
import numpy as np
src_obj = bpy.data.objects['source']
tar_obj = bpy.data.objects['target']

tar_mesh = tar_obj.data
ntar_verts = len(tar_mesh.vertices)
tar_kd = mathutils.kdtree.KDTree(ntar_verts)
for i,v in enumerate(tar_mesh.vertices):
    tar_kd.insert(v.co, i)
tar_kd.balance()

K = 10
src_mesh = src_obj.data
mappings = []
for idx, v in enumerate(src_mesh.vertices):
    results = tar_kd.find_n(v.co, n=K)
    #print(results)
    total_dst = 0.0
    for tup in results:
        total_dst += tup[2]
    v_maps = []
    for tup in results:
        w = tup[2]/total_dst
        w = np.exp(-w)
        v_maps.append((tup[1], w))
    mappings.append(v_maps)

#print(mappings)
out_path = "/media/khanhhh/42855ff5-574e-4a41-ad10-0f08087b0ff6/data_1/projects/Oh/codes/human_estimation/data/meta_data/alice_mpii_vert_verts_mapping.pkl"
with open(out_path, 'wb') as file:
    pickle.dump(file=file, obj=mappings)
    
    
    