import numpy as np
from scipy.interpolate import splprep, splev
import argparse
import os
import matplotlib.pyplot as plt
from pathlib import Path
from src.obj_util import load_vertices, export_vertices
from tsp_solver.greedy import solve_tsp
import pickle

def hack_fix_noise_vertices(slice):
    good_candidate_z = np.median(slice[:,2], axis=0)
    epsilon = 0.00001
    slice_1 = []
    noise_cnt = 0
    for i in range(slice.shape[0]):
        co = slice[i, :]
        if np.abs(co[2] - good_candidate_z) < epsilon:
            slice_1.append(co)
        else:
            noise_cnt+=1

    return np.array(slice_1), noise_cnt

def resample_slice(slice, n_point, debug_path = None):
    pts = slice[:,:2]

    centroid = np.mean(pts, axis=0)

    #sort clock wise
    D = np.zeros(shape=(pts.shape[0], pts.shape[0]), dtype=np.float32)
    for i in range(pts.shape[0]):
        dsts = np.linalg.norm(pts - pts[i,:], axis=1)
        D[i,:] =dsts
    path = solve_tsp(D, optim_steps=5000)
    pts_s = pts[path, :]
    xp = pts_s[:,0]
    yp = pts_s[:,1]

    #remove noise
    okay = np.where(np.abs(np.diff(xp)) + np.abs(np.diff(yp)) > 0)
    xp = np.r_[xp[okay], xp[-1], xp[0]]
    yp = np.r_[yp[okay], yp[-1], yp[0]]

    tck, u = splprep([xp, yp], u=None, s=0.0, per=1)
    u_new = np.linspace(u.min(), u.max(), n_point)
    x_new, y_new = splev(u_new, tck, der=0)

    slice_new = np.vstack([x_new, y_new, np.zeros_like(y_new)]).T
    slice_new[:,2] = np.median(slice[:,2])

    if debug_path is not None:
        centroid_new = np.array([np.mean(x_new), np.mean(y_new)])
        plt.clf()
        plt.plot(centroid[0], centroid[1], 'ro')
        plt.plot(centroid_new[0], centroid_new[1], 'yo')
        plt.plot(xp, yp, 'ro')
        plt.plot(x_new, y_new, 'yo')
        plt.plot(x_new, y_new, 'b--')
        plt.savefig(debug_path)

    return slice_new

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
           print(f'cannot map slice location {loc}', file=sys.stderr)

        z = loc[2]
        z_adjust = np.median(slices[closest_slice_id][:,2])
        tolerance = 0.01
        if np.abs(z-z_adjust) > tolerance:
            print(f'warning, something wrong. z slice location is larger than {tolerance}', file=sys.stderr)

        if closest_slice_id in loc_map:
            print(f'ERROR: something wrong. another location: {loc} is mapped to the same slice id: {closest_slice_id}', file=sys.stderr)

        print(f'adjust z location of slice {closest_slice_id}: z_old = {z}, z_adjust = {z_adjust}, delta = {np.abs(z-z_adjust)}')
        locs[i,2] = z_adjust

        loc_map[closest_slice_id] = locs[i,:]

    return loc_map

def is_a_leg_slice(name):
    leg_hints=['Knee', 'Thigh', 'Ankle', 'Calf']
    for hint in leg_hints:
        if hint in name:
            return True
    return False

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--in_dir", required=True, help="slice obj directory")
    ap.add_argument("-o", "--out_dir", required=True, help="directory for expxorting control mesh slices")
    args = vars(ap.parse_args())

    DIR_IN  = args['in_dir']
    DIR_OUT = args['out_dir']
    DIR_DEBUG = Path(os.path.abspath(__file__))
    DIR_DEBUG  = str(DIR_DEBUG.parent) + '/../data/debug/'

    for fpath in Path(DIR_OUT).glob('*.*'):
        os.remove(fpath)

    slices = {}
    for fpath in Path(DIR_IN).glob('*.obj'):
        if 'slice_location' in str(fpath):
            continue

        print(fpath)
        debug_path =  f'{DIR_DEBUG}{fpath.stem}.png'
        print(debug_path)
        #if 'L14_Shoulder' not in str(fpath):
        #    continue

        slice = load_vertices(fpath)
        slice, removed_cnt = hack_fix_noise_vertices(slice)
        if removed_cnt > 0:
            print(f'remove {removed_cnt} vertices: {fpath.stem}')

        if is_a_leg_slice(fpath.stem):
            resolution = 100*12
        else:
            resolution = 100*8
        slice = resample_slice(slice, n_point=resolution, debug_path=debug_path)
        assert slice.shape[0] == resolution

        slices[fpath.stem] = slice

        export_vertices(f'{DIR_OUT}/{fpath.name}', slice)

    for fpath in Path(DIR_IN).glob('*.pkl'):
        if 'victoria' in str(fpath):
            with open(fpath, 'rb') as f:
                data = pickle.load(f)

            slice_locs = data['slice_locs']
            loc_map = map_slice_location_to_slices(slices, slice_locs)
            data['slice_locs'] = slice_locs

            with open(f'{DIR_OUT}/{fpath.stem}.pkl', 'wb') as f:
                pickle.dump(data, f)





