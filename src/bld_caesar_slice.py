import bpy
import bmesh
from mathutils import Vector
from pathlib import Path
import pickle as pkl
import numpy as np
import os
from collections import defaultdict

scene = bpy.context.scene

def select_single_obj(obj):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select = True
    bpy.context.scene.objects.active = obj

#boolean will apply on obj1
def isect(obj1, obj2):
    select_single_obj(obj1)
    mod = obj1.modifiers.new('Boolean', type='BOOLEAN')
    mod.object = obj2
    mod.solver ='BMESH'
    mod.operation = 'INTERSECT'
    bpy.ops.object.modifier_apply(apply_as='DATA', modifier = mod.name)

def copy_obj(obj, name, location):
    select_single_obj(obj)
    bpy.ops.object.duplicate()
    obj = scene.objects.active
    obj.name = name
    return obj

def delete_obj(obj):
    select_single_obj(obj)
    bpy.ops.object.delete()

def import_obj(path, name):
    bpy.ops.import_scene.obj(filepath=path, axis_forward='-Y', axis_up='Z', split_mode='OFF')
    s = 0.01
    bpy.ops.transform.resize(value=(s,s,s))
    bpy.ops.transform.rotate(value=180.0, axis=(0.0,0.0,1.0))
    obj = bpy.context.selected_objects[0]
    obj.name = name
    select_single_obj(obj)
    bpy.ops.object.transform_apply(location=True, scale=True, rotation=True)
    return obj

def mesh_to_numpy(mesh):
    nverts = len(mesh.vertices)
    verts = []
    for iv in range(nverts):
        verts.append(mesh.vertices[iv].co[:])
    
    faces = []
    for p in mesh.polygons:
        faces.append(p.vertices[:])
    
    return np.array(verts), faces

def set_plane_slice_loc(planes, name, loc):
    mesh = planes.data
    found = False
    for v in mesh.vertices:
        assert len(v.groups) <= 1
        for vgrp in v.groups:
            grp_name = planes.vertex_groups[vgrp.group].name
            #print('group name', grp_name, name)
            if grp_name == name:
                v.co[2] = loc[2]
                found = True
    assert found

def calc_slice_plane_locs(cae_obj, ld_idxs):
    verts = cae_obj.data.vertices
    locs = {}
    loc_0 = verts[ld_idxs[16]].co
    loc_1 = verts[ld_idxs[18]].co
    locs['Hip'] =  0.5*(loc_0 + loc_1)
    
    loc_0 = verts[ld_idxs[12]].co
    loc_1 = verts[ld_idxs[13]].co
    loc_2 = verts[ld_idxs[14]].co
    locs['Bust'] = (loc_0+loc_1+loc_2)/3.0
    
    print('Hip', locs['Hip'])
    print('Bust',locs['Bust'])
    
    return locs

def fit_slice_planes_to_mesh(cae_obj, planes, ld_idxs):
    locs = calc_slice_plane_locs(cae_obj, ld_idxs)
    for id, loc in locs.items():
        set_plane_slice_loc(planes, id, loc)
    return locs

def extract_slice_from_isect_obj(locs, isect_obj, eps = 0.1):
    isect_bm = bmesh.new()
    isect_bm.from_mesh(isect_obj.data)
    
    slc_contours = defaultdict(list)
    f_marks = set()
    for id, loc in locs.items():
        for f in isect_bm.faces:
            center = f.calc_center_bounds()
            if abs(center[2] - loc[2]) < eps:
                #a face should belong to a single slice
                assert f not in f_marks
                f_marks.add(f)
                #extract face contour
                contour = []
                for v in f.verts:
                    contour.append(v.co[:])
                    
                slc_contours[id].append(contour)
        
        #assert that we found at least one contour for that slice
        assert id in slc_contours
                                               
    return slc_contours
    
if __name__ == '__main__':
    DIR_OBJ = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj/'
    
    DIR_SLICE = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj_slices/'
    
    ld_path = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_meta/landmarksIdxs73.npy'
    ld_idxs = np.load(ld_path)
    
    os.makedirs(DIR_SLICE, exist_ok=True)
    
    obj_planes_tpl = bpy.data.objects['Slice_planes_template']
    
    for i, path in enumerate(Path(DIR_OBJ).glob('*.obj')):
        if '_ld' not in str(path):
            print(str(path))
            obj_caesar = import_obj(str(path),'Caesar_mesh')
            ld_path = DIR_OBJ + path.stem+'_ld'+'.obj'
            obj_caesar_ld = import_obj(ld_path,'Caesar_mesh_ld')
            
            obj_planes = copy_obj(obj_planes_tpl, 'Slice_planes', Vector((0.0,0.0,0.0)))            
            slc_locs = fit_slice_planes_to_mesh(obj_caesar, obj_planes, ld_idxs)
            
            isect(obj_planes, obj_caesar)
            delete_obj(obj_caesar)
            delete_obj(obj_caesar_ld)
            
            #verts, faces = mesh_to_numpy(obj_planes.data)
            slc_contours = extract_slice_from_isect_obj(slc_locs, obj_planes)
            delete_obj(obj_planes)
            
            with open(str(Path(DIR_SLICE, path.stem+'.pkl')), 'wb') as file:
                pkl.dump(slc_contours, file)

            if i > 0:
                break            
    