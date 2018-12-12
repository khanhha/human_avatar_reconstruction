import bpy
import os
import pickle
import numpy as  np
import mathutils.geometry as geo
from mathutils import Vector
from bpy import context
from collections import defaultdict
from copy import deepcopy

def select_single_obj(obj):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select = True
    bpy.context.scene.objects.active = obj

def isect(obj1, obj2):
    select_single_obj(obj1)
    mod = obj1.modifiers.new('Boolean', type='BOOLEAN')
    mod.object = obj2
    mod.solver ='BMESH'
    mod.operation = 'INTERSECT'
    bpy.ops.object.modifier_apply(apply_as='DATA', modifier = mod.name)

def copy_obj(obj, new_name, location):
    obj_data = obj.data
    new_object = bpy.data.objects.new(name=new_name, object_data=obj_data)
    scene.objects.link(new_object)
    new_object.location = location
    select_single_obj(new_object)
    return new_object

def write_vertices_to_obj(path, mesh):
    with open(path, 'w') as f:
        for i,v in enumerate(mesh.vertices):
            f.write("v %.4f %.4f %.4f \n" % v.co[:])
        
def write_slices_to_obj(dir, obj, plane_z):
    mesh = obj.data
    print(plane_z)
    filepath = os.path.join(dir, obj.name+'.obj')
    write_vertices_to_obj(filepath, mesh)
            

def extract_vertices(obj):
    nverts = len(obj.data.vertices)
    arr = np.zeros((nverts, 3), np.float32)
    for i in range(nverts):
        arr[i,:] = obj.data.vertices[i].co[:]
    return arr        

def extract_vertices_by_face(obj):
    mesh = obj.data
    verts = []
    for p in mesh.polygons:
        for i in p.vertices:
            co = mesh.vertices[i].co[:]
            verts.append(co)
    return np.array(verts)

def mesh_to_numpy(mesh):
    nverts = len(mesh.vertices)
    verts = []
    for iv in range(nverts):
        verts.append(mesh.vertices[iv].co[:])
    
    faces = []
    for p in mesh.polygons:
        faces.append(p.vertices[:])
    
    return np.array(verts), faces

def slice_type(name):
    if ( 'Knee' in name) or ('Ankle' in name) or ('Thigh' in name) or ( 'Calf' in name) or ('Foot' in name) or ('UnderCrotch' in name):
        if name[0] == 'L':
            return 'LLEG'
        else:    
            return 'RLEG'
    elif 'Elbow' in name or 'Wrist' in name or 'Hand' in name: 
        return 'ARM'
    else:
        return 'TORSO'

def rect_plane(rect):
    point = np.mean(rect, axis=0)
    v0 = rect[2,:] - rect[0,:]
    v1 = rect[1,:] - rect[0,:]
    n = np.cross(v0, v1)
    return Vector(point), Vector(n).normalized()

def bone_location(arm_obj, bone, type):
    if type == 'head':
        return np.array((arm_obj.location + bone.head_local)[:])
    else:
        return np.array(( arm_obj.location + bone.tail_local)[:])
    
def extract_armature_bone_locations(obj):
    bones = obj.data.bones
    bdata = {}

    mappings = [
        ['LToe', 'foot.L', 'tail'],
        ['LHeel', 'heel.L', 'tail'],
        ['LAnkle', 'shin.L', 'tail'],
        ['LKnee', 'thigh.L', 'tail'],
        ['LHip', 'thigh.L', 'head'],
        ['Crotch', 'hips', 'head'],
        ['Spine', 'spine', 'head'],
        ['Chest', 'chest', 'head'],
        ['Neck', 'neck', 'head'],
        ['NeckEnd', 'neck', 'tail'],
        ['LShoulder', 'shoulder.L', 'tail'],
        ['LElbow', 'upper_arm.L', 'tail'],
        ['LWrist', 'forearm.L', 'tail'],
        ['LHand', 'hand.L', 'tail']]
    
    for m in mappings:
        bdata[m[0]] = bone_location(obj, bones[m[1]], m[2])
        if m[0][0] == 'L':
            m[0] = 'R' + m[0][1:]
            m[1] = m[1].replace('.L', '.R')        
            bdata[m[0]] = bone_location(obj, bones[m[1]], m[2])
            
    return bdata 
    
def slice_plane(mesh, slc_vert_idxs):
    centroid = Vector((0,0,0))
    n_vert = len(slc_vert_idxs)
    for v_idx in slc_vert_idxs:
        centroid += mesh.vertices[v_idx].co
    centroid /= float(n_vert)
    p0 = mesh.vertices[slc_vert_idxs[0]].co
    p1 = mesh.vertices[slc_vert_idxs[int(n_vert/2)]].co
    normal = (p1 - centroid).cross(p0-centroid)
    normal = normal.normalized()
    return centroid, normal
    
def calc_slice_location(amt_obj, ctl_obj, slc_vert_idxs):
    torso_bones = ['hips', 'spine', 'chest', 'neck', 'head']
    arm_bones = ['upper_arm.R', 'forearm.R', 'hand.R', 'upper_arm.L', 'forearm.L', 'hand.L']
    lleg_bones = ['thigh.L', 'shin.L', 'foot.L']
    rleg_bones = ['thigh.R', 'shin.R', 'foot.R']
    
    bones = amt_obj.data.bones
    ctl_mesh = ctl_obj.data
    
    slc_id_locs = {}
 
    for id, v_idxs in slc_vert_idxs.items():
        body_part =  slice_type(id)
        if body_part == 'ARM':
            bone_ids = arm_bones
        elif body_part == 'LLEG':   
            bone_ids = lleg_bones
        elif body_part == 'RLEG':
            bone_ids = rleg_bones
        else:
            bone_ids = torso_bones
        
        pln_p, pln_n = slice_plane(ctl_mesh, v_idxs)
        
        for bone_id in bone_ids:
            b = bones[bone_id]
            head = amt_obj.location +  b.head_local
            tail = amt_obj.location + b.tail_local
            isct_p = geo.intersect_line_plane(head, tail, pln_p, pln_n)
            if isct_p is not None:
                if (isct_p - head).dot(tail-head) < 0.0:
                    continue
                if (isct_p - head).length > (tail-head).length:
                    continue

                slc_id_locs[id] = np.array(isct_p[:])
                break
            
    for id, val in slc_vert_idxs.items():
        if id not in slc_id_locs:
            print('failed location: ', id)
        
    if (len(slc_id_locs.keys()) != len(slc_rects.items())):
        print('failed to find all slice locations')
        
    return slc_id_locs    

def body_part_dict():
    maps = {}
    maps['Part_LArm'] = 1
    maps['Part_RArm'] = 2
    maps['Part_LLeg'] = 3
    maps['Part_RLeg'] = 4
    maps['Part_Torso'] = 5
    maps['Part_Head']  = 6
    return maps

def extract_body_part_indices(obj, grp_mark):
    mesh = obj.data
    maps = body_part_dict()    
    v_types  = np.zeros(len(mesh.vertices), dtype=np.uint8)        
    cnt = 0
    for v in mesh.vertices:
        ##assert len(v.groups) == 1
        for vgrp in v.groups:
            grp_name = obj.vertex_groups[vgrp.group].name
            bd_part_name = grp_name.split('.')[0]
            assert bd_part_name in maps
            id  = maps[bd_part_name] 
            v_types[v.index] = id
            cnt += 1         
               
    assert cnt == len(mesh.vertices)
    
    return v_types
    
def extract_slice_vert_indices(ctl_obj):
    print(ctl_obj.name)
    mesh = ctl_obj.data
    slc_vert_idxs = defaultdict(list)
    for v in mesh.vertices:
        #assert len(v.groups) == 1
        for vgrp in v.groups:
                grp_name =  ctl_obj.vertex_groups[vgrp.group].name
                slc_vert_idxs[grp_name].append(v.index)
    output = {}
    for slc_id, idxs in slc_vert_idxs.items():
        output[slc_id] = np.array(idxs) 

    return output

def extract_body_part_face_indices(obj, grp_mark):
    mesh = obj.data
    n_faces = len(mesh.polygons)
    
    face_types = np.zeros(len(mesh.polygons), dtype=np.uint8)
    maps = body_part_dict()    

    for pol in mesh.polygons:
        mat_idx = pol.material_index
        mat_name = obj.material_slots[mat_idx].name
        #hack. ControlMesh_Tri is duplicated from ControlMesh
        #therefore, its material name often has postfix .00N
        bd_part_name = mat_name.split('.')[0]
        assert bd_part_name in maps 
        face_types[pol.index] = maps[bd_part_name]
        
    return face_types

def find_mirror_vertices(obj, group_name):
    mesh = obj.data
    my_verts = []
    my_verts_idxs = []
    for idx, v in enumerate(mesh.vertices):
        for vgrp in v.groups:
            grp_name = obj.vertex_groups[vgrp.group].name
            if grp_name == group_name:
                my_verts.append(v)
                my_verts_idxs.append(idx)

    assert len(my_verts) > 0

    mirror_idxs = []
    for mv in my_verts:
        mirror_co   = deepcopy(mv.co)
        mirror_co.x = -mirror_co.x
        dsts = np.array([(mirror_co-ov.co).length for ov in mesh.vertices])
        idx = np.argmin(dsts)
        mirror_idxs.append(idx)

    assert len(my_verts_idxs) == len(mirror_idxs)

    pairs = [(idx_0, idx_1) for idx_0, idx_1 in zip(my_verts_idxs, mirror_idxs)]

    return pairs

context = bpy.context
scene = context.scene

OUT_DIR = '/home/khanhhh/data_1/projects/Oh/data/bl_models/victoria_ctr_mesh/'


height_locs = extract_vertices(scene.objects['HeightSegment'])

slc_rects = {}
for obj in scene.objects:
    n_parts = len(obj.name.split('_'))
    if 'plane' in obj.name:  
        if len(obj.data.vertices) != 4: 
            print('something wrong', obj.name)

        pln_verts = extract_vertices_by_face(obj)
        id_3d = obj.name.replace('_plane','')
        slc_rects[id_3d] = pln_verts
        
arm_bone_locs = extract_armature_bone_locations(bpy.data.objects["Armature"])

slc_vert_idxs = extract_slice_vert_indices(bpy.context.scene.objects["ControlMesh"])

slc_id_locs = calc_slice_location(bpy.data.objects["Armature"], bpy.data.objects["ControlMesh_Tri"], slc_vert_idxs)
        
ctl_obj = scene.objects['ControlMesh_Tri']
ctl_verts, ctl_faces = mesh_to_numpy(ctl_obj.data)

ctl_obj_quad = scene.objects['ControlMesh']
ctl_verts_quad, ctl_faces_quad = mesh_to_numpy(ctl_obj_quad.data)

vic_obj = scene.objects['VictoriaMesh']
vic_verts, vic_faces = mesh_to_numpy(vic_obj.data)
print('victoria mesh: nverts = {0}, nfaces = {1}'.format(vic_verts.shape[0], len(vic_faces)))

vic_v_body_parts = extract_body_part_indices(vic_obj, grp_mark = 'Part_')
ctl_f_body_parts = extract_body_part_face_indices(ctl_obj, grp_mark = 'Part_')
print('classified {0} faces of control mesh to body part'.format(ctl_f_body_parts.shape[0]))
assert ctl_f_body_parts.shape[0] == len(ctl_obj.data.polygons)


mirror_pairs = find_mirror_vertices(bpy.data.objects['ControlMesh'], 'LBody')

filepath = os.path.join(OUT_DIR, 'victoria.pkl')
with open(filepath, 'wb') as f:
    data = {}
    data['slice_locs'] = slc_id_locs
    data['arm_bone_locs'] = arm_bone_locs
    data['height_segment'] = height_locs
    data['slice_vert_idxs'] = slc_vert_idxs

    data['mirror_pairs'] = mirror_pairs
    
    data['ctl_mesh'] = {'verts':ctl_verts, 'faces':ctl_faces}    
    data['ctl_f_body_parts'] = ctl_f_body_parts
    data['ctl_mesh_quad_dom'] = {'verts':ctl_verts_quad, 'faces':ctl_faces_quad}   
    
    data['vic_mesh'] = {'verts':vic_verts, 'faces':vic_faces}
    data['vic_v_body_parts'] = vic_v_body_parts     
    
    data['body_part_dict'] = {v:k for k,v in body_part_dict().items()}
    pickle.dump(data, f)


victoria = scene.objects['VictoriaMesh']    
for obj in scene.objects:
    n_parts = len(obj.name.split('_'))
    #if obj.name.split('_')[0][0] == 'L' and obj.name.split('_')[0][1].isdigit():  
        #plane_z = obj.data.vertices[0].co[2]
        #select_single_obj(obj)
        #isect(obj, victoria) 
        #write_slices_to_obj(OUT_DIR, obj, plane_z)
        #print(obj.name)3   