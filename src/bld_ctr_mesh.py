import bpy
import os
import pickle
import numpy as  np

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
            
def rename_slice():
    context = bpy.context
    scene = context.scene
    
    slice_planes = []
    for obj in scene.objects:
        if 'L_' in obj.name:
            slice_planes.append(obj)

    slice_planes = sorted(slice_planes, key=lambda obj: obj.location[2])
    for i, obj in enumerate(slice_planes):
        postfix = obj.name.split('_')[1]
        obj.name = 'L_'+str(i)+'_'+postfix

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
    
    return np.array(verts), np.array(faces)

import sys
def map_slice_location_to_slices(slices, locs):
    loc_map = {}
    
    for i in range(locs.shape[0]):
        loc = locs[i,:]
        closest_slice_id = None
        #find slices which has the closest z
        min_dst = 999999999
        for id, slice in slices.items():
            dst = np.linalg.norm(loc - np.mean(slice, axis=0))
            if dst < min_dst:
                min_dst = dst
                closest_slice_id = id

        if closest_slice_id is None:
           print('cannot map slice location {loc}', file=sys.stderr)

        z = loc[2]
        z_adjust = np.median(slices[closest_slice_id][:,2])
        tolerance = 0.01
        if np.abs(z-z_adjust) > tolerance:
            print('warning, something wrong. z slice location is larger than {tolerance}', file=sys.stderr)

        if closest_slice_id in loc_map:
            print('ERROR: something wrong. another location: {loc} is mapped to the same slice id: {closest_slice_id}', file=sys.stderr)
            
        #print(f'adjust z location of slice {closest_slice_id}: z_old = {z}, z_adjust = {z_adjust}, delta = {np.abs(z-z_adjust)}')
        locs[i,2] = z_adjust

        loc_map[closest_slice_id] = locs[i,:]
    
    return loc_map
        
context = bpy.context
scene = context.scene

OUT_DIR = '/home/khanhhh/data_1/projects/Oh/data/bl_models/victoria_ctr_mesh/'


height_locs = extract_vertices(scene.objects['HeightSegment'])

slc_rects = {}
for obj in scene.objects:
    n_parts = len(obj.name.split('_'))
    if obj.name.split('_')[0][0] == 'L' and obj.name.split('_')[0][1].isdigit():  
        if len(obj.data.vertices) != 4: continue
        print(obj.name)
        pln_verts = extract_vertices_by_face(obj)
        slc_rects[obj.name] = pln_verts
        
slc_locs    = extract_vertices(scene.objects['SliceLocation'])
slc_locs_map = map_slice_location_to_slices(slc_rects, slc_locs)

        
ctl_obj = scene.objects['ControlMesh']
ctl_verts, ctl_faces = mesh_to_numpy(ctl_obj.data)
        
filepath = os.path.join(OUT_DIR, 'victoria.pkl')
with open(filepath, 'wb') as f:
    data = {}
    data['slice_locs'] = slc_locs_map
    data['height_segment'] = height_locs
    data['slice_rects'] = slc_rects 
    data['ctl_mesh'] = {'verts':ctl_verts, 'faces':ctl_faces}    
    pickle.dump(data, f)


victoria = scene.objects['Figure_2_node_Figure_2_geometr']
for obj in scene.objects:
    n_parts = len(obj.name.split('_'))
    if obj.name.split('_')[0][0] == 'L' and obj.name.split('_')[0][1].isdigit():  
        plane_z = obj.data.vertices[0].co[2]
        #select_single_obj(obj)
        #isect(obj, victoria) 
        #write_slices_to_obj(OUT_DIR, obj, plane_z)
        #print(obj.name)


        
    
