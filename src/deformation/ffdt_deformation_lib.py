import numpy as np
from collections import defaultdict
from common.util_math import normalize
import multiprocessing
from functools import partial
import sys
from copy import deepcopy
from scipy import stats
from scipy.spatial import KDTree
from tqdm import tqdm

#section 3.1, "t-FFD: Free-Form Deformation by using Triangular Mesh"
def parameterize(verts, vert_effect_idxs, basis):

    def parallel_util(start, end, out_queue, out_dir):
        P_ret = []
        T = np.zeros((3, 3), np.float32)
        #print(f'\tstart range {start}-{end}')
        for i in range(start, end):
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

            P_ret.append((i, v_uvw))

        out_path = f'{out_dir}/{start}_{end}.pkl'

        with open(out_path,'wb') as file:
            pickle.dump(obj=P_ret, file=file)

        out_queue.put(out_path)

        #print(f'\t\tfinish range {start}-{end}')

    N = verts.shape[0]
    nprocess = 12
    result_queue = multiprocessing.Queue()
    step = N//12
    procs = []
    for i in range(nprocess):
        start = i*step
        end = (i+1)*step
        if i == nprocess-1 and end != N:
            end = N
        p = Process(target=parallel_util, args=(start, end, result_queue, tempfile.gettempdir()))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    P = N*[None]
    cnt = 0
    #print('start merging data')
    for i in range(nprocess):
        data_path = result_queue.get()
        with open(data_path, 'rb') as file:
            pro_data = pickle.load(file)

            for pair in pro_data:
                idx = pair[0]
                v_uvw = pair[1]
                P[idx] = v_uvw
                cnt += 1

        os.remove(data_path)

    assert cnt == N, 'failed to calculate all vertices'
    for data in P:
        assert data != None, 'missing vertex'
    return P

#section 3.1, "t-FFD: Free-Form Deformation by using Triangular Mesh"
def calc_triangle_local_basis(verts, tris):
    # basis = np.zeros((len(tris),4, 3), dtype=np.float32)
    # for i, t in enumerate(tris):
    #     if len(t) != 3:
    #         raise Exception(f' face {i} is not a triangle. unable to calculate triangle bases')
    #     basis[i, 0, :] = verts[t[0]]
    #     basis[i, 1, :] = normalize(verts[t[1]] - verts[t[0]])
    #     basis[i, 2, :] = normalize(verts[t[2]] - verts[t[0]])
    #     basis[i, 3, :] = normalize(np.cross(basis[i, 1, :], basis[i, 2, :]))

    tris= np.array(tris)
    b_0 = verts[tris[:,1]] - verts[tris[:,0]]
    b_0 = (b_0.T / np.linalg.norm(b_0, axis=1)).T

    b_1 = verts[tris[:,2]] - verts[tris[:,0]]
    b_1 = (b_1.T / np.linalg.norm(b_1, axis=1)).T

    b_2 = np.cross(b_0, b_1, axisa=1, axisb=1)
    b_2 = (b_2.T / np.linalg.norm(b_2, axis=1)).T

    # diff_1 = np.abs(basis[:,1,:] - b_0)
    # diff_2 = np.abs(basis[:,2,:] - b_1)
    # diff_3 = np.abs(basis[:,3,:] - b_2)

    basis = np.zeros((len(tris),4, 3), dtype=np.float32)
    basis[:, 0, :] = verts[tris[:,0]]
    basis[:, 1, :] = b_0
    basis[:, 2, :] = b_1
    basis[:, 3, :] = b_2

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
                    w = (1 - w)**2
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



from multiprocessing import Process
import tempfile
import os
def calc_vert_weight(tpl_verts, vert_effect_tri_idxs, ctl_verts, ctl_tris, effective_range_factor):

    def parallel_util(start, end, result_queue, out_dir):
        #print(f'\tstart range {start}-{end}')
        ret = []
        for v_idx in range(start, end):
            # if v_idx != 23383:
            #     continue
            v = tpl_verts[v_idx, :]
            tri_idxs = vert_effect_tri_idxs[v_idx]
            weights = []
            new_tri_idxs = []
            for t_idx in tri_idxs:
                t = ctl_tris[t_idx]
                v0, v1, v2 = ctl_verts[t[0], :], ctl_verts[t[1],:], ctl_verts[t[2],:]
                center = (v0 + v1 + v2) / 3.0
                radius = (np.linalg.norm(v0 - center) + np.linalg.norm(v1 - center) + np.linalg.norm(v2 - center)) / 3.0
                d = np.linalg.norm(v - center)
                ratio = d / radius
                if ratio < effective_range_factor:
                    w = ratio / effective_range_factor
                    w = 1.0 - w
                    #w = np.exp(-w)
                    new_tri_idxs.append(t_idx)
                    weights.append(w)

            ret.append((v_idx, new_tri_idxs, weights))

        #print(f'\t\tfinish range {start}-{end}')
        out_path = f'{out_dir}/{start}_{end}.pkl'
        with open(out_path, 'wb') as file:
            pickle.dump(file=file, obj=ret)
        result_queue.put(out_path)
        #print(f'\t\tdumped result of range {start}-{end} to file {out_path}')

    N = len(vert_effect_tri_idxs)
    nprocess = 12
    result_queue = multiprocessing.Queue()
    step = N//12
    procs = []
    for i in range(nprocess):
        start = i*step
        end = (i+1)*step
        if i == nprocess-1 and end != N:
            end = N
        p = Process(target=parallel_util, args=(start, end, result_queue, tempfile.gettempdir()))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    new_vert_effect_tri_idxs = N*[None]
    verts_weights = N*[None]
    cnt = 0
    for i in range(nprocess):
        path = result_queue.get()
        with open(path, 'rb') as file:
            p_tuples = pickle.load(file)
            for tuple in p_tuples:
                new_vert_effect_tri_idxs[tuple[0]] = tuple[1]
                verts_weights[tuple[0]] = tuple[2]
                cnt += 1
        os.remove(path)
    assert cnt == N
    print(f'\nfinish weight calculation for {cnt} vertices')
    return new_vert_effect_tri_idxs, verts_weights

    # for v_idx, tri_idxs in enumerate(vert_effect_tri_idxs):
    #     if v_idx % 1000 == 0:
    #         print(v_idx)
    #
    #     v = tpl_verts[v_idx, :]
    #     weights = []
    #     new_tri_idxs = []
    #     for t_idx in tri_idxs:
    #         t = ctl_tris[t_idx]
    #         v0, v1, v2 = ctl_verts[t[0]], ctl_verts[t[1]], ctl_verts[t[2]]
    #         center = (v0 + v1 + v2) / 3.0
    #         radius = (np.linalg.norm(v0-center) + np.linalg.norm(v1-center) + np.linalg.norm(v2-center))/3.0
    #         d = np.linalg.norm(v - center)
    #         ratio = d / radius
    #         if ratio < effective_range_factor:
    #             w = ratio/effective_range_factor
    #             w = 1.0 - w
    #             #w = np.exp(-w)
    #             new_tri_idxs.append(t_idx)
    #             weights.append(w)
    #
    #     new_vert_effect_tri_idxs.append(new_tri_idxs)
    #     verts_weights.append(weights)
    #
    # return new_vert_effect_tri_idxs, verts_weights


def calc_vertex_weight_global(v, tri_centers, tri_kd_tree, tri_radius, effective_range_factor, query_radius):

    neighbor_tri_idxs = tri_kd_tree.query_ball_point(v, query_radius)

    idxs = []
    weights = []

    for j in neighbor_tri_idxs:
        d = np.linalg.norm(v - tri_centers[j, :])
        ratio = d/tri_radius[j]

        if ratio < effective_range_factor:
            w = ratio/effective_range_factor
            w = 1.0 - w
            #w = np.exp(w*w)
            idxs.append(j)
            weights.append(w)

    if len(idxs) == 0:
        print(f'isolated vertex: {v}')
        dsts, neighbor_tri_idxs = tri_kd_tree.query(v, k = 45)
        cls_dst = dsts[0]
        min_effect_range_factor = cls_dst / neighbor_tri_idxs[0]
        total_d = 0.0
        for j in neighbor_tri_idxs:
            d = np.linalg.norm(v - tri_centers[j, :])
            total_d += d

        for j in neighbor_tri_idxs:
            d = np.linalg.norm(v - tri_centers[j, :])
            ratio = d/tri_radius[j]
            w = np.exp(-ratio)
            idxs.append(j)
            weights.append(w)

    assert len(idxs) > 0, 'isolated points. no weights are calculated'

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


    tri_kd_tree = KDTree(tri_centers)
    query_radius = tri_radius.max() * effective_range_factor
    nprocess = n_process
    pool = multiprocessing.Pool(nprocess)

    results = pool.map(func=partial(calc_vertex_weight_global,
                                    tri_kd_tree = tri_kd_tree, tri_centers=tri_centers, tri_radius =tri_radius, effective_range_factor=effective_range_factor, query_radius=query_radius),
                       iterable=verts, chunksize=128)
    for pair in results:
        effect_idxs.append(pair[0])
        effect_weights.append(pair[1])

    return effect_idxs, effect_weights

#section 3.4 Mapping, "t-FFD: Free-Form Deformation by using Triangular Mesh"
def deform_template_mesh(vert_tri_idxs, vert_weights, vert_UVWs, ctl_df_basis):
    # for i in deform_vert_idxs:
    #     tri_idxs    =   vert_tri_idxs[i]
    #     tri_basis   =   ctl_df_basis[tri_idxs,:,:]
    #     tri_uvws    =   vert_UVWs[i]
    #     tri_weights =   vert_weights[i]
    #     b_0 = (tri_basis[:,1, :].T * tri_uvws[:,0]).T
    #     b_1 = (tri_basis[:,2, :].T * tri_uvws[:,1]).T
    #     b_2 = (tri_basis[:,3, :].T * tri_uvws[:,2]).T
    #     coords = tri_basis[:,0,:] + b_0 + b_1 + b_2
    #     df_co_1 = np.average(coords, weights=tri_weights, axis=0)
    #     df_verts[i, :] = df_co_1
    #
    # diff = np.abs(df_coords - df_verts)
    #return df_verts

    v_b_0 = ctl_df_basis[vert_tri_idxs,1,:] #12894x107x3
    v_w_0 = vert_UVWs[:, :, 0] #12894x107
    v_b_0 = v_b_0 * np.expand_dims(v_w_0, axis=2)

    v_b_1 = ctl_df_basis[vert_tri_idxs,2,:] #12894x107x3
    v_w_1 = vert_UVWs[:, :, 1] #12894x107
    v_b_1 = v_b_1 * np.expand_dims(v_w_1, axis=2)

    v_b_2 = ctl_df_basis[vert_tri_idxs,3,:] #12894x107x3
    v_w_2 = vert_UVWs[:, :, 2] #12894x107
    v_b_2 = v_b_2 * np.expand_dims(v_w_2, axis=2)

    v_org = ctl_df_basis[vert_tri_idxs, 0, :]

    coords = v_org + v_b_0 + v_b_1 + v_b_2

    v_weights = np.dstack([vert_weights, vert_weights, vert_weights])
    df_coords = np.average(coords, weights=v_weights, axis=1)

    return df_coords

import pickle
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

    def set_parameterization(self, vert_tri_UVWs, vert_tri_weights, vert_effect_tri_idxs):
        n_max_neighbour = np.max([len(vert_tri_UVWs[i]) for i in range(len(vert_tri_UVWs))])
        n_v = len(vert_tri_UVWs)

        #padding all vertex information to the size of n_max_neighbor for the sake of parallerization
        for i in range(n_v):
            v_tri_idx = vert_effect_tri_idxs[i]
            v_tri_uvw = vert_tri_UVWs[i]
            v_tri_weigths = vert_tri_weights[i]

            cur_len = len(v_tri_idx)
            assert cur_len == len(v_tri_uvw)
            assert cur_len == len(v_tri_weigths)

            pad = n_max_neighbour - cur_len

            v_tri_idx = v_tri_idx + pad*[v_tri_idx[0]]
            v_tri_uvw = v_tri_uvw + pad*[v_tri_uvw[0]]
            v_tri_weigths = v_tri_weigths + pad*[0.0] #very important. zero weight padding so padded elements has no effect

            vert_effect_tri_idxs[i] = v_tri_idx
            vert_tri_UVWs[i] = v_tri_uvw
            vert_tri_weights[i] = v_tri_weigths

        self.vert_tri_UVWs    =    np.array(vert_tri_UVWs)
        self.vert_tri_weights =    np.array(vert_tri_weights)
        self.vert_tri_idxs    =    np.array(vert_effect_tri_idxs)

    def calculate_parameterization(self):
        print(f'\nstart calculating local basis (U,V,W) for each control mesh triangle \m')
        self.ctl_tri_bs =  calc_triangle_local_basis(self.ctl_verts, self.ctl_tris)
        print(f'\tfinish local basis calculation')

        print(f'\nstart calculating weights')
        print(f'\n\teffective range = {self.effective_range}, use_mean_radius = {self.use_mean_rad}')
        self.vert_tri_idxs, self.vert_tri_weights = calc_vertex_weigth_control_mesh_global(self.tpl_verts, self.ctl_verts, self.ctl_tris,
                                                                                           effective_range_factor = self.effective_range,
                                                                                           use_mean_tri_radius = self.use_mean_rad)
        lens = np.array([len(idxs) for idxs in self.vert_tri_idxs])
        stat = stats.describe(lens)
        print(f'\tfinish weight calculation')
        print(f'\tneighbor size statistics: mean number of neighbor, variance number of neighbor')
        print(f'\t{stat}')

    def deform(self, new_ctl_vert):
        if new_ctl_vert.shape[0] != self.ctl_verts.shape[0]:
            raise RuntimeError('invalid input control vertex array')

        ctl_new_basis = calc_triangle_local_basis(new_ctl_vert, self.ctl_tris)

        tpl_new_verts = deform_template_mesh(self.vert_tri_idxs, self.vert_tri_weights, self.vert_tri_UVWs, ctl_new_basis)

        return tpl_new_verts