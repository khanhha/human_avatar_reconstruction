import numpy as np
from scipy.interpolate import splprep, splev
import argparse
import os
import matplotlib.pyplot as plt
from pathlib import Path
from src.obj_util import load_slice_template_from_obj_file, export_slice_obj
from shapely.geometry import MultiPoint
from tsp_solver.greedy import solve_tsp

def hack_fix_noise_vertices(slice):
    idx = int(slice.shape[0]/2)
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

def clockwiseangle_and_distance(point, org):
    refvec = np.array([0, 1])
    # Vector between point and the origin: v = p - o
    vector = [point[0]-org[0], point[1]-org[1]]
    # Length of vector: ||v||
    lenvector = np.hypot(vector[0], vector[1])
    # If length is zero there is no angle
    if lenvector == 0:
        return -np.pi, 0

    # Normalize vector: v/||v||
    normalized = [vector[0]/lenvector, vector[1]/lenvector]

    dotprod  = normalized[0]*refvec[0] + normalized[1]*refvec[1]     # x1*x2 + y1*y2
    diffprod = refvec[1]*normalized[0] - refvec[0]*normalized[1]     # x1*y2 - y1*x2
    angle = np.arctan2(diffprod, dotprod)

    # Negative angles represent counter-clockwise angles so we need to subtract them
    # from 2*pi (360 degrees)
    if angle < 0:
        return 2*np.pi+angle, lenvector

    # I return first the angle because that's the primary sorting criterium
    # but if two vectors have the same angle then the shorter distance should come first.
    return angle, lenvector

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
        print(fpath)
        debug_path =  f'{DIR_DEBUG}{fpath.stem}.png'
        print(debug_path)
        #if 'L14_Shoulder' not in str(fpath):
        #    continue

        slice = load_slice_template_from_obj_file(fpath)
        slice, removed_cnt = hack_fix_noise_vertices(slice)
        if removed_cnt > 0:
            print(f'remove {removed_cnt} vertices: {fpath.stem}')
        slice = resample_slice(slice, n_point=50, debug_path=debug_path)
        export_slice_obj(f'{DIR_OUT}/{fpath.name}', slice)