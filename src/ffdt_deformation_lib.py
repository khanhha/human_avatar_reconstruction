import numpy as np
from collections import defaultdict
from util_math import normalize
import multiprocessing
from functools import partial
import sys
from obj_util import import_mesh
from copy import deepcopy
from scipy import stats

#section 3.1, "t-FFD: Free-Form Deformation by using Triangular Mesh"
def parameterize(verts, vert_effect_idxs, basis):
    P = []
    T = np.zeros((3,3),np.float32)
    for i in range(verts.shape[0]):
        v_co = verts[i,:]
        idxs = vert_effect_idxs[i]
        v_uvw = []
        for ctl_tri_idx in idxs:
            d = v_co - basis[ctl_tri_idx, 0, :]
            T[:,0] = basis[ctl_tri_idx, 1, :]
            T[:,1] = basis[ctl_tri_idx, 2, :]
            T[:,2] = basis[ctl_tri_idx, 3, :]
            try:
                local_co = np.linalg.solve(T, d)
                v_uvw.append(local_co)
            except Exception as exp:
                print(f'exception: {exp}', file=sys.stderr)
                print(f'linear system A = {T}')
                print(f'right hand side = {d}')
                exit()
        P.append(v_uvw)

        #if i % 500 == 0:
        #    print(i,'/',verts.shape[0])

    return P

#section 3.1, "t-FFD: Free-Form Deformation by using Triangular Mesh"
def calc_triangle_local_basis(verts, tris):
    basis = np.zeros((len(tris),4, 3), dtype=np.float32)
    for i, t in enumerate(tris):
        if len(t) != 3:
            raise Exception(f' face {i} is not a triangle. unable to calculate triangle bases')
        basis[i, 0, :] = verts[t[0]]
        basis[i, 1, :] = verts[t[1]] - verts[t[0]]
        basis[i, 2, :] = verts[t[2]] - verts[t[0]]
        basis[i, 3, :] = normalize(np.cross(basis[i, 1, :], basis[i, 2, :]))
    return basis

def average_circumscribed_radius_tris(verts, tris, tris_idxs):
    rads = np.zeros(len(tris_idxs), dtype=np.float32)
    for j, t_idx in enumerate(tris_idxs):
        v0, v1, v2 = verts[tris[t_idx][0]], verts[tris[t_idx][1]], verts[tris[t_idx][2]]
        center = (v0+v1+v2)/3.0
        rads[j] = (np.linalg.norm(v0-center) + np.linalg.norm(v1-center) + np.linalg.norm(v2-center))/3.0
    return np.mean(rads)

def is_adjacent_tri(t0, t1):
    for i in range(3):
        v00, v01 = t0[i], t0[(i+1)%3]
        for j in range(3):
            v10, v11 = t1[j], t1[(j+1)%3]
            if (v00 == v10 and v01 == v11) or (v00 == v11 and v01 == v10):
                return True
    return False

def find_neighbour_body_part_triangles(tris_ctl, ctl_tri_bd_parts):
    n_tris = len(tris_ctl)
    bd_part_adj_tri_idxs = defaultdict(list)

    for tri_idx, part_id in enumerate(ctl_tri_bd_parts):
        for other_tri_idx in range(n_tris):
            #not the triangle we are processing
            if tri_idx == other_tri_idx:
                continue

            #not in the same body part
            if ctl_tri_bd_parts[other_tri_idx] == part_id:
                continue

            if is_adjacent_tri(tris_ctl[tri_idx], tris_ctl[other_tri_idx]):
                bd_part_adj_tri_idxs[part_id].append(other_tri_idx)

    return bd_part_adj_tri_idxs

def calc_vertex_weigth_control_mesh_local(verts_tpl, verts_ctl, tris_ctl, tpl_v_body_parts, ctl_tri_bd_parts, effective_range_factor = 6):

    effect_idxs = [None]* len(verts_tpl)
    effect_weights =[None]* len(verts_tpl)

    ctl_tri_centers = np.zeros((len(tris_ctl), 3), dtype=np.float32)
    for j, t in enumerate(tris_ctl):
        v0, v1, v2 = verts_ctl[t[0]], verts_ctl[t[1]], verts_ctl[t[2]]
        center = (v0+v1+v2)/3.0
        ctl_tri_centers[j, :] = center

    bd_part_ctl_tris_idxs = defaultdict(list)
    for tri_idx, _ in enumerate(tris_ctl):
        part_id = ctl_tri_bd_parts[tri_idx]
        bd_part_ctl_tris_idxs[part_id].append(tri_idx)

    print('generating neighbouring information for control mesh')
    bd_part_adj_ctl_tri_idxs = find_neighbour_body_part_triangles(tris_ctl, ctl_tri_bd_parts)

    bd_part_ctl_avg_rads = {id : average_circumscribed_radius_tris(verts_ctl, tris_ctl, part_tris_idxs) for id, part_tris_idxs in bd_part_ctl_tris_idxs.items()}
    print(bd_part_ctl_avg_rads )

    body_part_tpl_verts = {}
    for i, part_id in enumerate(tpl_v_body_parts):
        if part_id not in body_part_tpl_verts:
            body_part_tpl_verts[part_id] = []
        body_part_tpl_verts[part_id].append(i)

    print('start calculating weights')
    cnt_progress = 0
    n_zero_weights = 0
    for part_id, verts  in body_part_tpl_verts.items():
        if part_id not in bd_part_ctl_avg_rads:
            print(f'missing body part of ID {part_id}')
            continue
        print(f'processing body part {part_id}')
        bd_part_avg_rad = bd_part_ctl_avg_rads[part_id]
        for v_i in  verts:
            tri_idxs = []
            weights = []
            v_co = verts_tpl[v_i, :]

            #weights respect to control tringle in the same body part
            ctl_tris_idxs = bd_part_ctl_tris_idxs[part_id]
            for tri_idx in ctl_tris_idxs:
                d = np.linalg.norm(v_co - ctl_tri_centers[tri_idx, :])
                ratio = d / bd_part_avg_rad
                if ratio < effective_range_factor:
                    w = ratio / effective_range_factor
                    w = 1 - w
                    #w = np.exp(-4.0*w*w)
                    tri_idxs.append(tri_idx)
                    weights.append(w)

            #weights respect to neighbour control triangle in the adjacent body part
            ctl_tris_idxs = bd_part_adj_ctl_tri_idxs[part_id]
            for tri_idx in ctl_tris_idxs:
                d = np.linalg.norm(v_co - ctl_tri_centers[tri_idx, :])
                ratio = d / bd_part_avg_rad
                if ratio < effective_range_factor:
                    w = 1.0 - ratio / effective_range_factor
                    tri_idxs.append(tri_idx)
                    weights.append(w)

            cnt_progress += 1
            if len(tri_idxs) == 0:
                n_zero_weights += 1
            if cnt_progress % 500 == 0:
                print(cnt_progress, '/', verts_tpl.shape[0], 'zero weight vertices = ', n_zero_weights)

            effect_idxs[v_i] = tri_idxs
            effect_weights[v_i] = weights

    return effect_idxs, effect_weights

def calc_vertex_weight_global(v, tris_ref, tri_centers, tri_radius, effective_range_factor):
    idxs = []
    weights = []
    for j, t in enumerate(tris_ref):
        #w = vert_tri_weight(v, verts_ref[t[0]], verts_ref[t[1]], verts_ref[t[2]])
        d = np.linalg.norm(v - tri_centers[j, :])
        ratio = d/tri_radius[j]
        if ratio < effective_range_factor:
            #w = 1.0 - ratio / effective_range_factor
            w = ratio/effective_range_factor
            w = 1.0 - w
            #w = np.exp(w*w)
            idxs.append(j)
            weights.append(w)

    return (idxs, weights)

#section 3.2, "t-FFD: Free-Form Deformation by using Triangular Mesh"
def calc_vertex_weigth_control_mesh_global(verts, verts_ref, tris_ref, effective_range_factor = 4, use_mean_tri_radius = False, n_process = 12):
    effect_idxs = []
    effect_weights = []

    tri_centers = np.zeros((len(tris_ref), 3), dtype=np.float32)
    tri_radius = np.zeros((len(tris_ref)), dtype=np.float32)
    for j, t in enumerate(tris_ref):
        v0, v1, v2 = verts_ref[t[0]], verts_ref[t[1]], verts_ref[t[2]]
        center = (v0+v1+v2)/3.0
        tri_centers[j, :] = center
        tri_radius[j] = (np.linalg.norm(v0-center) + np.linalg.norm(v1-center) + np.linalg.norm(v2-center))/3.0

    if use_mean_tri_radius:
        avg_rad = np.mean(tri_radius)
        tri_radius[:] = avg_rad

    nprocess = n_process
    pool = multiprocessing.Pool(nprocess)
    results = pool.map(func=partial(calc_vertex_weight_global, tris_ref=tris_ref, tri_centers=tri_centers, tri_radius =tri_radius, effective_range_factor=effective_range_factor),
                       iterable=verts, chunksize=128)

    for pair in results:
        effect_idxs.append(pair[0])
        effect_weights.append(pair[1])

    return effect_idxs, effect_weights

#section 3.4 Mapping, "t-FFD: Free-Form Deformation by using Triangular Mesh"
def deform_template_mesh(df_verts, effect_vert_tri_idxs, vert_weights, vert_UVWs, ctl_df_basis, deform_vert_idxs = None):
    nv = len(df_verts)

    #apply all or just a subset of vertices
    if deform_vert_idxs == None:
        deform_vert_idxs = range(nv)

    for i in deform_vert_idxs:
        df_co = np.zeros(3, np.float32)
        W = 0.0
        for idx, ev_idx in enumerate(effect_vert_tri_idxs[i]):
            df_basis = ctl_df_basis[ev_idx,:,:]
            uvw = vert_UVWs[i][idx]
            co = df_basis[0,:] + uvw[0]*df_basis[1,:]+ uvw[1]*df_basis[2,:]+ uvw[2]*df_basis[3,:]
            w_tri = vert_weights[i][idx]
            df_co += w_tri * co
            W += w_tri
        if W > 0:
            df_co /= W
            df_verts[i,:] = df_co

    return df_verts

class TemplateMeshDeform():

    def __init__(self, effective_range, use_mean_rad):
        self.effective_range = effective_range
        self.use_mean_rad = use_mean_rad

    def set_meshes(self, ctl_verts, ctl_tris, tpl_verts, tpl_faces):
        self.tpl_verts = tpl_verts
        self.tpl_faces = tpl_faces
        self.ctl_verts = ctl_verts
        self.ctl_tris  = ctl_tris
        for tri in self.ctl_tris:
            if len(tri) != 3:
                raise Exception('Non-triangle control mesh')

    def set_parameterization(self, ctl_tri_basis, vert_UVWs, vert_weights, vert_effect_idxs):
        self.ctl_tri_basis = ctl_tri_basis
        self.vert_UVWs = vert_UVWs
        self.vert_weights = vert_weights
        self.vert_effect_idxs = vert_effect_idxs

    def calculate_parameterization(self):
        print(f'\nstart calculating local basis (U,V,W) for each control mesh triangle \m')
        self.ctl_tri_bs =  calc_triangle_local_basis(self.ctl_verts, self.ctl_tris)
        print(f'\tfinish local basis calculation')

        print(f'\nstart calculating weights')
        print(f'\n\teffective range = {self.effective_range}, use_mean_radius = {self.use_mean_rad}')
        self.vert_effect_idxs, self.vert_weights = calc_vertex_weigth_control_mesh_global(self.tpl_verts, self.ctl_verts, self.ctl_tris,
                                                                                   effective_range_factor = self.effective_range,
                                                                                   use_mean_tri_radius = self.use_mean_rad)
        lens = np.array([len(idxs) for idxs in self.vert_effect_idxs])
        stat = stats.describe(lens)
        print(f'\tfinish weight calculation')
        print(f'\tneighbor size statistics: mean number of neighbor, variance number of neighbor')
        print(f'\t{stat}')

    def deform(self, new_ctl_vert):
        if new_ctl_vert.shape[0] != self.ctl_verts.shape[0]:
            raise RuntimeError('invalid input control vertex array')

        ctl_new_basis = calc_triangle_local_basis(new_ctl_vert, self.ctl_tris)

        tpl_new_verts = deepcopy(self.tpl_verts)
        tpl_new_faces = deepcopy(self.tpl_faces)
        deform_template_mesh(tpl_new_verts, self.vert_effect_idxs, self.vert_weights, self.vert_UVWs, ctl_new_basis)

        return tpl_new_verts, tpl_new_faces