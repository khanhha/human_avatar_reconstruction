import bpy
import bmesh
from mathutils import Vector
from pathlib import Path
import pickle as pkl
import numpy as np
import math
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
    bpy.ops.import_scene.obj(filepath=path, axis_forward='Y', axis_up='Z', split_mode='OFF')
    obj = bpy.context.selected_objects[0]
    obj.name = name
    return obj

def transform_obj_caesar(obj, ld_idxs):
    mesh = obj.data
    
    s = 0.01
    bpy.ops.transform.resize(value=(s,s,s))
    bpy.ops.object.transform_apply(location=True, scale=True, rotation=True)
    
    org = mesh.vertices[ld_idxs[72]].co
    bpy.ops.transform.translate(value = -org)
    bpy.ops.object.transform_apply(location=True, scale=False, rotation=False)
       
    p0_x = mesh.vertices[ld_idxs[16]].co
    p1_x = mesh.vertices[ld_idxs[18]].co
    x = p1_x - p0_x
    x.normalize()
    angle = x.dot(Vector((1.0, 0.0, 0.0)))
    #angle =  math.degrees(math.acos(angle))
    angle = math.acos(angle)
    print('angle', angle)
    bpy.ops.transform.rotate(value=angle, axis=(0.0,0.0,1.0))
    
    select_single_obj(obj)
    bpy.ops.object.transform_apply(location=True, scale=True, rotation=True)

def ucsc_transform_obj(obj):
    mesh = obj.data
    select_single_obj(obj)
        
    s = 5.0
    bpy.ops.transform.resize(value=(s,s,s))
    bpy.ops.object.transform_apply(location=True, scale=True, rotation=True)
    
    #org = mesh.vertices[ld_idxs[72]].co
    #bpy.ops.transform.translate(value = -org)
    #bpy.ops.object.transform_apply(location=True, scale=False, rotation=False)
       
    angle = math.radians(50.0)
    bpy.ops.transform.rotate(value=angle, axis=(0.0,0.0,1.0))
    bpy.ops.object.transform_apply(location=True, scale=True, rotation=True)
    
def mesh_to_numpy(mesh):
    nverts = len(mesh.vertices)
    verts = []
    for iv in range(nverts):
        verts.append(mesh.vertices[iv].co[:])
    
    faces = []
    for p in mesh.polygons:
        faces.append(p.vertices[:])
    
    return np.array(verts), faces

def ucsc_collect_landmark_vertices(obj):
    verts = obj.data.vertices
    lds = defaultdict(list)
    for v in verts:
        for vgrp in v.groups:
            grp_name = obj.vertex_groups[vgrp.group].name
            lds[grp_name].append(v.index)
    return lds
    
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

def centroid_vertex_set(vertices, ids):
    avg = Vector((0.0, 0.0, 0.0))
    for id in ids:
        v = vertices[id]
        avg = avg + v.co
    avg = avg / float(len(ids))
    return avg
    
def ucsc_calc_slice_plane_locs(cae_obj, ld_idxs):
    verts = cae_obj.data.vertices
    locs = {}
    
    bust_ids = ld_idxs['Bust']
    locs['Bust'] = centroid_vertex_set(verts, bust_ids)
    
    hip_ids = ld_idxs['Hip']
    locs['Hip'] = centroid_vertex_set(verts, hip_ids)
    
    return locs
    
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
    
    #print('Hip', locs['Hip'])
    #print('Bust',locs['Bust'])
    
    return locs

def fit_slice_planes_to_mesh(cae_obj, planes, ld_idxs):
    locs = calc_slice_plane_locs(cae_obj, ld_idxs)
    for id, loc in locs.items():
        set_plane_slice_loc(planes, id, loc)
    return locs

def ucsc_fit_slice_planes_to_mesh(cae_obj, planes, ld_idxs):
    locs = ucsc_calc_slice_plane_locs(cae_obj, ld_idxs)
    for id, loc in locs.items():
        set_plane_slice_loc(planes, id, loc)
    return locs

#should be in edit mode
def contour_from_edges(edges):
    contours = []
    bpy.ops.mesh.select_mode(type="EDGE")
    edge_mark = set(edges)
    while True:
        if len(edge_mark) == 0:
            break

        #select a single loop from a sample edge
        bpy.ops.mesh.select_all(action='DESELECT')
        prev_e = edge_mark.pop()
        prev_e.select = True
        bpy.ops.mesh.loop_multi_select()

        prev_e.select = False

        #sort all edges on this loop
        contour = []
        contour.append(prev_e.verts[0].co[:])
        contour.append(prev_e.verts[1].co[:])
        v_start = prev_e.verts[0]
        v_cur = prev_e.verts[1]
        while True:
            e_next = []
            for e in v_cur.link_edges:
                if e.is_valid and e != prev_e and e.select == True:
                    e_next.append(e)

            assert len(e_next)  <= 1
            if len(e_next) > 1:
                #bpy.ops.mesh.select_all(action='DESELECT')
                #print('len e_next ',len(e_next))
                #for test_e in e_next:
                #     test_e.select = True
                #v_start.select = True
                #v_cur.select = True 
                return None
                
            #no closed contour found
            if len(e_next) == 0:
                break

            e_next = e_next[0]
            e_next.select = True
            v_cur = e_next.verts[1] if v_cur == e_next.verts[0] else e_next.verts[0]

            #discard this edge from our edge pool
            edge_mark.discard(e_next)

            #finish a contour
            if v_cur == v_start:
                break

            prev_e = e_next

            contour.append(v_cur.co[:])

        contours.append(contour)

    return contours

#mesh must be in edit mode
def isect_slice_plane_obj(bm, co, no):
    bpy.ops.mesh.select_all(action='SELECT')
    #after this function, all isect edges will be selected
    bpy.ops.mesh.bisect(plane_co=co, plane_no = no)
    edges = [e for e in bm.edges if e.select == True]
    assert len(edges) > 0
    return contour_from_edges(edges)

def isect_extract_slice_from_locs(locs, obj_caesar):
    no = Vector((0.0,0.0,1.0))
    
    select_single_obj(obj_caesar)
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj_caesar.data)

    slc_contours = defaultdict(list)
    for id, loc in locs.items():
        contours = isect_slice_plane_obj(bm, loc, no)
        if contours is None:
            return None
        
        slc_contours[id] = contours

    bpy.ops.object.mode_set(mode='OBJECT')

    return slc_contours

def extract_slice_from_isect_obj(locs, isect_obj, eps = 0.1):    
    select_single_obj(isect_obj)
    
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.edge_face_add()
    
    isect_bm = bmesh.from_edit_mesh(isect_obj.data)
    
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
    
    bpy.ops.object.mode_set(mode='OBJECT')                               
    return slc_contours

def mpii_process(DIR_OBJ):
    obj_planes_tpl = bpy.data.objects['Slice_planes_template']
    
    ld_path = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_meta/landmarksIdxs73.npy'
    ld_idxs = np.load(ld_path)
    
    error_obj_paths = []
    
    for i, path in enumerate(Path(DIR_OBJ).glob('*.obj')):
        
        #if 'CSR0441A' not in str(path):
        #    continue
        
        if '_ld' not in str(path):
            print(str(path))
            obj_caesar = import_obj(str(path),'Caesar_mesh')
            transform_obj_caesar(obj_caesar, ld_idxs)
            
            n_verts = len(obj_caesar.data.vertices)  
            n_faces = len(obj_caesar.data.polygons)
            if n_verts != 6449 or n_faces != 12894:
                print('error object: ', path.stem)                
                error_obj_paths.append(str(path))
                delete_obj(obj_caesar)
                delete_obj(obj_planes)
                continue
            
            ld_path = DIR_OBJ + path.stem+'_ld'+'.obj'
            #obj_caesar_ld = import_obj(ld_path,'Caesar_mesh_ld')
            
            obj_planes = copy_obj(obj_planes_tpl, 'Slice_planes', Vector((0.0,0.0,0.0)))            
            slc_locs = fit_slice_planes_to_mesh(obj_caesar, obj_planes, ld_idxs)

            slc_contours = isect_extract_slice_from_locs(slc_locs, obj_caesar)
            assert slc_contours is not None

            delete_obj(obj_caesar)
            #delete_obj(obj_caesar_ld)
            delete_obj(obj_planes)
            
            with open(str(Path(DIR_SLICE, path.stem+'.pkl')), 'wb') as file:
                pkl.dump(slc_contours, file)

            #if i > 100:
            #    break            
    
    print('error list: ', error_obj_paths)
    
def ucsc_process(DIR_OBJ, DIR_SLICE):
    obj_planes_tpl = bpy.data.objects['Slice_planes_template']
    
    error_obj_paths = []
    obj_template = bpy.data.objects['Female_template']
    ld_idxs = ucsc_collect_landmark_vertices(obj_template)

    for i, path in enumerate(Path(DIR_OBJ).glob('*.obj')):
        print(str(path))
        obj_caesar = import_obj(str(path), path.stem)
        
        select_single_obj(obj_caesar)
        bpy.ops.mesh.customdata_custom_splitnormals_clear()

        ucsc_transform_obj(obj_caesar)
    
        n_verts = len(obj_caesar.data.vertices)  
        n_faces = len(obj_caesar.data.polygons)
        if n_verts != 12500 or n_faces != 24999:
            print('error object: ', path.stem)                
            error_obj_paths.append(str(path))
            delete_obj(obj_caesar)
            continue
                
        obj_planes = copy_obj(obj_planes_tpl, path.stem+"_pln", Vector((0.0,0.0,0.0)))            
        obj_planes.hide = False
        obj_caesar.hide = False
        if i > 0:
            break    
        continue
    
        slc_locs = ucsc_fit_slice_planes_to_mesh(obj_caesar, obj_planes, ld_idxs)

    
        slc_contours = isect_extract_slice_from_locs(slc_locs, obj_caesar)
        assert slc_contours is not None

        delete_obj(obj_caesar)
        delete_obj(obj_planes)
        
        with open(str(Path(DIR_SLICE, path.stem+'.pkl')), 'wb') as file:
            pkl.dump(slc_contours, file)

        if i > 1:
            break            
    
    print('error list: ', error_obj_paths)
    
if __name__ == '__main__':
    #DIR_OBJ = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj/'
    #DIR_SLICE = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj_slices/'
    
    DIR_OBJ = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_usce/SPRING_FEMALE/'
    DIR_SLICE = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_usce/female_slice/'
    
    os.makedirs(DIR_SLICE, exist_ok=True)
    
    ucsc_process(DIR_OBJ, DIR_SLICE)
    
