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

def select_single_obj(obj):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select = True
    bpy.context.scene.objects.active = obj
    

def isect(obj1, obj2):
    select_single_obj(obj1)
    mod = obj1.modifiers.new('Boolean', type='BOOLEAN')
    mod.object = obj2
    mod.solver = 'BMESH'
    mod.operation = 'INTERSECT'
    bpy.ops.object.modifier_apply(apply_as='DATA', modifier=mod.name)


def copy_obj(obj, new_name, location):
    obj_data = obj.data
    new_object = bpy.data.objects.new(name=new_name, object_data=obj_data)
    scene.objects.link(new_object)
    new_object.location = location
    select_single_obj(new_object)
    bpy.ops.object.make_single_user(type='SELECTED_OBJECTS', object=True, obdata=True)
    return new_object


def export_mesh_to_obj(fpath, verts, faces, add_one=True):
    with open(fpath, 'w') as f:
        for i in range(verts.shape[0]):
            co = tuple(verts[i, :])
            f.write("v %.8f %.8f %.8f \n" % co)

        for i in range(len(faces)):
            f.write("f")
            for v_idx in faces[i]:
                if add_one == True:
                    v_idx += 1
                f.write(" %d" % (v_idx))
            f.write("\n")


def extract_vertices(obj):
    nverts = len(obj.data.vertices)
    arr = np.zeros((nverts, 3), np.float32)
    for i in range(nverts):
        arr[i, :] = obj.data.vertices[i].co[:]
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

    edges = []
    for e in mesh.edges:
        edges.append(e.vertices[:])

    v_e = [[] for _ in range(nverts)]
    nedges = len(mesh.edges)
    for ie in range(nedges):
        e = mesh.edges[ie]
        iv_0 = e.vertices[0]
        iv_1 = e.vertices[1]
        v_e[iv_0].append(ie)
        v_e[iv_1].append(ie)

    v_f = [[] for _ in range(nverts)]
    nfaces = len(mesh.polygons)
    for ip in range(nfaces):
        p = mesh.polygons[ip]
        for iv in p.vertices:
            v_f[iv].append(ip)

    e_f = [[] for _ in range(nedges)]
    for ip in range(nfaces):
        p = mesh.polygons[ip]
        for il in p.loop_indices:
            l = mesh.loops[il]
            e_f[l.edge_index].append(ip)

    loops = [(l.vertex_index, l.edge_index) for l in mesh.loops]
    f_l = [[] for _ in range(nfaces)]
    for ip in range(nfaces):
        p = mesh.polygons[ip]
        for il in p.loop_indices:
            f_l[ip].append(il)

    mesh = {'verts': np.array(verts), 'faces': faces, 'edges': edges, 'loops': loops, 'f_l': f_l, 'v_e': v_e,
            'v_f': v_f, 'e_f': e_f}

    return mesh


def slice_type(name):
    if ('Knee' in name) or ('Ankle' in name) or ('Thigh' in name) or ('Calf' in name) or ('Foot' in name) or (
            'UnderCrotch' in name):
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
    v0 = rect[2, :] - rect[0, :]
    v1 = rect[1, :] - rect[0, :]
    n = np.cross(v0, v1)
    return Vector(point), Vector(n).normalized()


def bone_location(arm_obj, bone, type):
    if type == 'head':
        return np.array((arm_obj.location + bone.head_local)[:])
    else:
        return np.array((arm_obj.location + bone.tail_local)[:])


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
    centroid = Vector((0, 0, 0))
    n_vert = len(slc_vert_idxs)
    for v_idx in slc_vert_idxs:
        centroid += mesh.vertices[v_idx].co
    centroid /= float(n_vert)
    p0 = mesh.vertices[slc_vert_idxs[0]].co
    p1 = mesh.vertices[slc_vert_idxs[int(n_vert / 2)]].co
    normal = (p1 - centroid).cross(p0 - centroid)
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
        body_part = slice_type(id)
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
            head = amt_obj.location + b.head_local
            tail = amt_obj.location + b.tail_local
            isct_p = geo.intersect_line_plane(head, tail, pln_p, pln_n)
            if isct_p is not None:
                if (isct_p - head).dot(tail - head) < 0.0:
                    continue
                if (isct_p - head).length > (tail - head).length:
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
    maps['Part_Head'] = 6
    return maps


def extract_body_part_indices(obj, grp_mark):
    mesh = obj.data
    maps = body_part_dict()
    v_types = np.zeros(len(mesh.vertices), dtype=np.uint8)
    for v in mesh.vertices:
        ##assert len(v.groups) == 1
        for vgrp in v.groups:
            grp_name = obj.vertex_groups[vgrp.group].name
            bd_part_name = grp_name.split('.')[0]
            if bd_part_name in maps:
                id = maps[bd_part_name]
                v_types[v.index] = id
    return v_types


def clockwiseangle_and_distance(point, org):
    refvec = np.array([0, 1])
    # Vector between point and the origin: v = p - o
    vector = [point[0] - org[0], point[1] - org[1]]
    # Length of vector: ||v||
    lenvector = np.hypot(vector[0], vector[1])
    # If length is zero there is no angle
    if lenvector == 0:
        return -np.pi, 0

    # Normalize vector: v/||v||
    normalized = [vector[0] / lenvector, vector[1] / lenvector]

    dotprod = normalized[0] * refvec[0] + normalized[1] * refvec[1]  # x1*x2 + y1*y2
    diffprod = refvec[1] * normalized[0] - refvec[0] * normalized[1]  # x1*y2 - y1*x2
    angle = np.arctan2(diffprod, dotprod)

    # Negative angles represent counter-clockwise angles so we need to subtract them
    # from 2*pi (360 degrees)
    if angle < 0:
        return 2 * np.pi + angle, lenvector

    # I return first the angle because that's the primary sorting criterium
    # but if two vectors have the same angle then the shorter distance should come first.
    return angle, lenvector


def arg_sort_points_cw(points):
    center = (np.mean(points[:, 0]), np.mean(points[:, 1]))
    compare_func = lambda pair: clockwiseangle_and_distance(pair[1], center)
    points = sorted(enumerate(points), key=compare_func)
    return [pair[0] for pair in points[::-1]]


def sort_leg_slice_vertices(slc_vert_idxs, mesh_verts):
    X = mesh_verts[slc_vert_idxs][:, 1]
    Y = mesh_verts[slc_vert_idxs][:, 0]

    org_points = np.concatenate([X[:, np.newaxis], Y[:, np.newaxis]], axis=1)

    points_0 = np.concatenate([X[:, np.newaxis], Y[:, np.newaxis]], axis=1)
    arg_points_0 = arg_sort_points_cw(points_0)
    points_0 = np.array(points_0[arg_points_0, :])

    # find the starting point of the leg contour
    # the contour must start at that point to match the order of the prediction contour
    # check the leg contour in the blender file for why it is this way
    start_idx = np.argmin(points_0[:, 1]) + 3
    points_0 = np.roll(points_0, axis=0, shift=-int(start_idx))

    # concatenate two sorted part.
    sorted_points = points_0

    # map indices
    slc_sorted_vert_idxs = []
    for i in range(sorted_points.shape[0]):
        p = sorted_points[i, :]
        dsts = np.sum(np.square(org_points - p), axis=1)
        closest_idx = np.argmin(dsts)
        found_idx = slc_vert_idxs[closest_idx]
        #make sure that we don't have duplicate
        assert found_idx not in slc_sorted_vert_idxs
        slc_sorted_vert_idxs.append(found_idx)

    # sorted_X =  mesh_verts[slc_sorted_vert_idxs][:,0]
    # sorted_Y =  mesh_verts[slc_sorted_vert_idxs][:,1]
    # plt.clf()
    # plt.axes().set_aspect(1)
    # plt.plot(points_0[:,0], points_0[:,1], '+r')
    # plt.plot(sorted_points[:5,0], sorted_points[:5,1],'-b')
    # plt.plot(sorted_X, sorted_Y,'-r')
    # plt.show()
    return slc_sorted_vert_idxs


def sort_torso_slice_vertices(slc_vert_idxs, mesh_verts, title=''):
    X = mesh_verts[slc_vert_idxs][:, 1]
    Y = mesh_verts[slc_vert_idxs][:, 0]

    org_points = np.concatenate([X[:, np.newaxis], Y[:, np.newaxis]], axis=1)

    # we needs to split our point array into two part because the clockwise sort just works on convex polygon. it will fail at strong concave points at crotch slice
    # sort the upper part
    mask_0 = Y >= -0.01
    X_0 = X[mask_0]
    Y_0 = Y[mask_0]
    assert (len(X_0) > 0 and len(Y_0) > 0)
    points_0 = np.concatenate([X_0[:, np.newaxis], Y_0[:, np.newaxis]], axis=1)
    arg_points_0 = arg_sort_points_cw(points_0)
    print("\t\tpart one: ", arg_points_0)
    points_0 = np.array(points_0[arg_points_0, :])

    # find the first point of the contour
    # the contour must start at that point to match the order of the prediction contour
    # check the leg contour in the blender file for why it is this way
    min_y = np.inf
    min_y_idx = 0
    for i in range(points_0.shape[0]):
        if points_0[i, 0] > 0:
            if points_0[i, 1] < min_y:
                min_y = points_0[i, 1]
                min_y_idx = i
    points_0 = np.roll(points_0, axis=0, shift=-min_y_idx)

    # sort the below part
    mask_1 = ~mask_0
    X_1 = X[mask_1]
    Y_1 = Y[mask_1]
    assert (len(X_1) > 0 and len(Y_1) > 0)
    points_1 = np.concatenate([X_1[:, np.newaxis], Y_1[:, np.newaxis]], axis=1)
    arg_points_1 = arg_sort_points_cw(points_1)
    print("\t\tpart two: ", arg_points_1)
    points_1 = np.array(points_1[arg_points_1, :])

    # concatenate two sorted part.
    sorted_points = np.concatenate([points_0, points_1], axis=0)
    #print("sorted points")
    #print(sorted_points)
    #print("org points")
    #print(org_points)
    #print("slc_vert_idxs")
    #print(slc_vert_idxs)
    # map indices
    slc_sorted_vert_idxs = []
    #print("mapping points")
    for i in range(sorted_points.shape[0]):
        p = sorted_points[i, :]
        dsts = np.sum(np.square(org_points - p), axis=1)
        closest_idx = np.argmin(dsts)
        found_idx = slc_vert_idxs[closest_idx]
        assert found_idx not in slc_sorted_vert_idxs
        slc_sorted_vert_idxs.append(found_idx)

    # sorted_X =  mesh_verts[slc_sorted_vert_idxs][:,0]
    # sorted_Y =  mesh_verts[slc_sorted_vert_idxs][:,1]
    # plt.clf()
    # plt.axes().set_aspect(1)
    # plt.plot(points_0[:,0], points_0[:,1], '+r')
    # plt.plot(sorted_points[:,0], sorted_points[:,1],'-b')
    # plt.plot(sorted_X, sorted_Y,'-r')
    # plt.title(title)
    # plt.show()
    return slc_sorted_vert_idxs


def is_torso_slice(id):
    torso_slc_ids = {'Crotch', 'Aux_Crotch_Hip_0', 'Aux_Crotch_Hip_1', 'Aux_Crotch_Hip_2', 'Hip', 'Waist', 'UnderBust',
                     'Aux_Hip_Waist_0', 'Aux_Hip_Waist_1',
                     'Aux_Waist_UnderBust_0', 'Aux_Waist_UnderBust_1','Aux_Waist_UnderBust_2',
                     'Aux_UnderBust_Bust_0', 'Bust', 'Armscye', 'Aux_Armscye_Shoulder_0', 'Shoulder'}
    if id in torso_slc_ids:
        return True
    else:
        return False


def is_leg_slice(id):
    leg_slc_ids = {'Knee', 'UnderCrotch', 'Aux_Knee_UnderCrotch_3',
                   'Aux_Knee_UnderCrotch_2','Aux_Knee_UnderCrotch_1', 'Aux_Knee_UnderCrotch_0'}
    if id in leg_slc_ids:
        return True
    else:
        return False

def extract_slice_vert_indices(ctl_obj):
    print(ctl_obj.name)
    mesh = ctl_obj.data

    mdata = mesh_to_numpy(mesh)
    ctl_verts = mdata['verts']

    slc_vert_idxs = defaultdict(list)
    for v in mesh.vertices:
        # assert len(v.groups) == 1
        for vgrp in v.groups:
            grp_name = ctl_obj.vertex_groups[vgrp.group].name
            slc_vert_idxs[grp_name].append(v.index)
    output = {}
    for slc_id, idxs in slc_vert_idxs.items():
        output[slc_id] = np.array(idxs)

    print('sortinng slice vertices counter clockwise, starting from the extreme point on the +X axis')
    for id, slc_idxs in output.items():
        if is_leg_slice(id):
            print('\t\t sort slice: ', id)
            slc_idxs = sort_leg_slice_vertices(slc_idxs, ctl_verts)
            output[id] = slc_idxs
        elif is_torso_slice(id):
            print('\t\t sort slice: ', id)
            slc_idxs = sort_torso_slice_vertices(slc_idxs, ctl_verts, title=id)
            output[id] = slc_idxs
        else:
            print('\t\t ignore slice: ', id)

    return output


def extract_body_part_face_indices(obj, grp_mark):
    mesh = obj.data

    face_types = np.zeros(len(mesh.polygons), dtype=np.uint8)
    maps = body_part_dict()

    for pol in mesh.polygons:
        mat_idx = pol.material_index
        mat_name = obj.material_slots[mat_idx].name
        # hack. ControlMesh_Tri is duplicated from ControlMesh
        # therefore, its material name often has postfix .00N
        bd_part_name = mat_name.split('.')[0]
        assert bd_part_name in maps
        face_types[pol.index] = maps[bd_part_name]

    return face_types

def find_mirror_vertices(obj, group_name, error_threshold=1e-3):
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
    pairs = [(idx_0, idx_1) for idx_0, idx_1 in zip(my_verts_idxs, mirror_idxs)]

    return pairs

def find_effective_cdd_triangles(tpl_mesh, ctl_mesh):
    print('start finding effective candidate control triangles for each vertex of Victoria')
    ctl_bm = bmesh.new()
    ctl_bm.from_mesh(ctl_mesh)

    ctl_bm.faces.ensure_lookup_table()
    ctl_bm.verts.ensure_lookup_table()

    n_ring = 2
    bvh = mathutils.bvhtree.BVHTree.FromBMesh(ctl_bm)

    cdd_tris = []
    for idx, v in enumerate(tpl_mesh.vertices):
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
        #print('found ', len(cur_faces), 'faces for vertex ', idx)
        cdd_tris.append(list(cur_faces))
    
    return cdd_tris

def triangulate_obj(obj):
    tri_obj_name = obj.name + '_Tri'

    if tri_obj_name in bpy.data.objects:
        old_obj_tri = bpy.data.objects[tri_obj_name]
        select_single_obj(old_obj_tri)
        bpy.ops.object.delete()

    obj_tri = copy_obj(obj, tri_obj_name, obj.location)
    select_single_obj(obj_tri)

    bpy.ops.object.modifier_add(type='TRIANGULATE')
    obj_tri.modifiers['Triangulate'].quad_method = 'BEAUTY'
    for modifier in obj_tri.modifiers:
        bpy.ops.object.modifier_apply(modifier=modifier.name)

    return obj_tri

scene = bpy.context.scene

OUT_DIR = '/home/khanhhh/data_1/projects/Oh/codes/human_estimation/data/meta_data/'

slc_rects = {}
for obj in scene.objects:
    n_parts = len(obj.name.split('_'))
    if 'plane' in obj.name:
        if len(obj.data.vertices) != 4:
            print('something wrong', obj.name)

        pln_verts = extract_vertices_by_face(obj)
        id_3d = obj.name.replace('_plane', '')
        slc_rects[id_3d] = pln_verts

arm_bone_locs = extract_armature_bone_locations(bpy.data.objects["Armature"])

slc_vert_idxs = extract_slice_vert_indices(bpy.context.scene.objects["ControlMesh"])

slc_id_locs = calc_slice_location(bpy.data.objects["Armature"], bpy.data.objects["ControlMesh"], slc_vert_idxs)

ctl_obj_tri = triangulate_obj(scene.objects['ControlMesh'])
ctl_obj_tri_mesh = mesh_to_numpy(ctl_obj_tri.data)

ctl_obj_quad = scene.objects['ControlMesh']
ctl_obj_quad_mesh = mesh_to_numpy(ctl_obj_quad.data)

vic_obj = scene.objects['VictoriaMesh']
vic_obj_mesh = mesh_to_numpy(vic_obj.data)

cdd_tris = find_effective_cdd_triangles(vic_obj.data, ctl_obj_tri.data)

vic_v_body_parts = extract_body_part_indices(vic_obj, grp_mark='Part_')
ctl_f_body_parts = extract_body_part_face_indices(ctl_obj_tri, grp_mark='Part_')
print('classified {0} faces of control mesh to body part'.format(ctl_f_body_parts.shape[0]))
assert ctl_f_body_parts.shape[0] == len(ctl_obj_tri.data.polygons)

height_locs = extract_vertices(scene.objects['HeightSegment'])

mirror_pairs = find_mirror_vertices(bpy.data.objects['ControlMesh'], 'LBody')

vic_mirror_pairs = find_mirror_vertices(bpy.data.objects['VictoriaMesh'], 'LBody')

filepath = os.path.join(OUT_DIR, 'vic_data.pkl')
print('output all data to file ', filepath)

cdd_tris_path = os.path.join(OUT_DIR, 'tpl_ctl_effective_cdd_tris.pkl')
with open(cdd_tris_path, 'wb') as f:
    pickle.dump(file=f, obj=cdd_tris)
        
with open(filepath, 'wb') as f:
    data = {}
    data['slice_locs'] = slc_id_locs
    data['slice_vert_idxs'] = slc_vert_idxs
    data['arm_bone_locs'] = arm_bone_locs
    
    data['control_mesh_symmetric_vert_pairs'] = mirror_pairs

    data['control_mesh'] = ctl_obj_tri_mesh
    data['control_mesh_face_body_parts'] = ctl_f_body_parts
    data['control_mesh_quad_dom'] = ctl_obj_quad_mesh

    data['template_mesh'] = vic_obj_mesh
    data['template_height'] = np.linalg.norm(height_locs)
    data['template_vert_body_parts'] = vic_v_body_parts
    data['template_symmetric_vert_pairs'] = vic_mirror_pairs
    data['body_part_dict'] = {v: k for k, v in body_part_dict().items()}
    pickle.dump(data, f)

# output the two meshes for the sake of debugging
filepath = os.path.join(OUT_DIR, 'origin_control_mesh_tri.obj')
print('output triangulated control mesh to ', filepath)
export_mesh_to_obj(filepath, ctl_obj_tri_mesh['verts'], ctl_obj_tri_mesh['faces'])

filepath = os.path.join(OUT_DIR, 'origin_template_mesh.obj')
print('output triangulated control mesh to ', filepath)
export_mesh_to_obj(filepath, vic_obj_mesh['verts'], vic_obj_mesh['faces'])