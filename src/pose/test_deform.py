import numpy as np
import sys
sys.path.insert(0, '../../third_parties/libigl/python/')
import pyigl as igl
import cv2 as cv
import matplotlib.pyplot as plt
from shapely.geometry import MultiPoint
from shapely.ops import triangulate
from scipy.spatial import Delaunay
import triangle as tr
def grab_contours(cnts):
    # if the length the contours tuple returned by cv2.findContours
    # is '2' then we are using either OpenCV v2.4, v4-beta, or
    # v4-official
    if len(cnts) == 2:
        cnts = cnts[0]

    # if the length of the contours tuple is '3' then we are using
    # either OpenCV v3, v4-pre, or v4-alpha
    elif len(cnts) == 3:
        cnts = cnts[1]

    # otherwise OpenCV has changed their cv2.findContours return
    # signature yet again and I have no idea WTH is going on
    else:
        raise Exception(("Contours tuple must have length 2 or 3, "
            "otherwise OpenCV changed their cv2.findContours return "
            "signature yet again. Refer to OpenCV's documentation "
            "in that case"))

    # return the actual contours array
    return cnts

from scipy.ndimage import filters
def smooth_contour(contour, sigma=3):
    contour_new = contour.astype(np.float32)
    contour_new[:, 0, 0] = filters.gaussian_filter1d(contour[:, 0, 0], sigma=sigma)
    contour_new[:, 0, 1] = filters.gaussian_filter1d(contour[:, 0, 1], sigma=sigma)
    return contour_new.astype(np.int32)

def contour_length(contour):
    n_point = contour.shape[0]
    l = 0.0
    for i in range(contour.shape[0]):
        i_nxt = (i + 1) % n_point
        l += np.linalg.norm(contour[i,0,:] - contour[i_nxt, 0,:])
    return l

def resample_contour(contour, n_keep_point):
    n_point = contour.shape[0]
    new_contour = np.zeros((n_keep_point, 1, 2), dtype=np.int32)
    cnt_len = contour_length(contour)
    step_len  = cnt_len / float(n_keep_point)
    acc_len = 0.0
    cur_idx = 0
    for i in range(1, n_point):
        p       = contour[i,0,:]
        p_prev  = contour[i-1,0,:]
        cur_e_len = np.linalg.norm(p-p_prev)
        if acc_len + cur_e_len >= step_len:
            residual = cur_e_len - (step_len - acc_len)
            inter_p = p_prev + (1-(residual/cur_e_len))* (p - p_prev)
            new_contour[cur_idx,0,:]  = inter_p
            cur_idx += 1
            acc_len = residual
        else:
            acc_len += cur_e_len

    for i in range(cur_idx, n_keep_point):
        new_contour[i,0,:] = contour[n_point-1, 0, :]

    return new_contour

def find_largest_contour(img_bi, app_type=cv.CHAIN_APPROX_TC89_L1):
    contours = cv.findContours(img_bi, cv.RETR_LIST, app_type)
    contours = grab_contours(contours)
    largest_cnt = None
    largest_area = -1
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > largest_area:
            largest_area = area
            largest_cnt = cnt
    return largest_cnt

def remove_outside_triangles(verts, triangles, img):
    centers = 0.3333*(verts[triangles[:,0], :] + verts[triangles[:,1], :]+verts[triangles[:,2], :])
    centers = centers.astype(np.uint32)
    centers_inside = img[centers[:,1],centers[:,0]] > 0

    new_triangles = triangles[centers_inside, :]
    new_vert_idxs = np.unique(new_triangles[:])
    new_verts = verts[new_vert_idxs, :]
    vmap = dict((new_vert_idxs[i], i) for i in range(new_vert_idxs.shape[0]))

    map_vidx_func = lambda old_idx : vmap[old_idx]
    new_triangles = np.vectorize(map_vidx_func)(new_triangles)

    return new_verts, new_triangles

def solve_biharmonic(verts, triangles, handle_idxs, moved_handles):
    V = igl.eigen.MatrixXd()
    V_bc = igl.eigen.MatrixXd()
    U_bc = igl.eigen.MatrixXd()

    F = igl.eigen.MatrixXi()
    b = igl.eigen.MatrixXi()

    n_bdr = len(handle_idxs)
    n_verts = verts.shape[0]
    n_tris = triangles.shape[0]

    F.resize(n_tris, 3)
    for i in range(n_tris):
        for k in range(3):
            F[i, k] = triangles[i][k]

    V.resize(n_verts, 2)
    for i in range(n_verts):
        for k in range(2):
            V[i,k] = verts[i,k]

    b.resize(n_bdr, 1)
    for i in range(n_bdr):
        b[i, 0] = handle_idxs[i]

    U_bc.resize(b.rows(), V.cols())
    V_bc.resize(b.rows(), V.cols())

    for i in range(n_bdr):
        for k in range(2):
            V_bc[i, k] = verts[handle_idxs[i], k]

    for i in range(n_bdr):
        for k in range(2):
            U_bc[i,k] = moved_handles[i, k]

    D = igl.eigen.MatrixXd()
    D_bc = U_bc - V_bc
    igl.harmonic(V, F, b, D_bc, 2, D)
    U = V + D

    verts_1 = np.copy(verts)
    for i in range(n_verts):
        for k in range(2):
            verts_1[i, k] = U[i,k]

    return verts_1

def solve_head_discontinuity(org_head_verts, embedded_head_verts, handle_idxs, head_tris_in_head):
    V = igl.eigen.MatrixXd()
    V_bc = igl.eigen.MatrixXd()
    U_bc = igl.eigen.MatrixXd()

    F = igl.eigen.MatrixXi()
    b = igl.eigen.MatrixXi()

    n_bdr = len(handle_idxs)
    n_verts = org_head_verts.shape[0]
    n_tris = len(head_tris_in_head)

    f.resize(len(head_tris_in_head), 3)
    for i in range(n_tris):
        assert len(head_tris_in_head[i]) == 3
        for k in range(3):
            F[i, k] = head_tris_in_head[i][k]

    V.resize(n_verts, 3)
    for i in range(n_verts):
        for k in range(3):
            V[i,k] = org_head_verts[i,k]

    b.resize(n_bdr, 1)
    for i in range(n_bdr):
        b[i, 0] = handle_idxs[i]

    U_bc.resize(b.rows(), V.cols())
    V_bc.resize(b.rows(), V.cols())

    for i in range(n_bdr):
        for k in range(3):
            V_bc[i, k] = org_head_verts[handle_idxs[i], k]

    for i in range(n_bdr):
        for k in range(3):
            U_bc[i,k] = embedded_head_verts[handle_idxs[i], k]

    D = igl.eigen.MatrixXd()
    D_bc = U_bc - V_bc
    igl.harmonic(V, F, b, D_bc, 2, D)
    U = V + D

    verts_1 = np.copy(org_head_verts)
    for i in range(len(org_head_verts)):
        for k in range(3):
            verts_1[i, k] = U[i,k]

    return verts_1

if __name__ == '__main__':
    # def circle(N, R):
    #     i = np.arange(N)
    #     theta = i * 2 * np.pi / N
    #     pts = np.stack([np.cos(theta), np.sin(theta)], axis=1) * R
    #     seg = np.stack([i, i + 1], axis=1) % N
    #     return pts, seg
    #
    #
    # pts0, seg0 = circle(30, 1.4)
    # pts1, seg1 = circle(16, 0.6)
    # #pts = np.vstack([pts0, pts1])
    # #seg = np.vstack([seg0, seg1 + seg0.shape[0]])
    # pts = pts0
    # seg = seg0
    #
    # A = dict(vertices=pts, segments=seg, holes=[[1.4, 0]])
    # B = tr.triangulate(A, 'qpa')
    # tr.compare(plt, A, B)
    # plt.show()


    names = dir(igl)
    dir ='/home/khanhhh/data_1/projects/Oh/data/3d_human/test_data/body_designer_result_nosyn/'
    img_path = f'{dir}/front_IMG_3869_sil.jpg'
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    img = (img > 0).astype(np.uint8)
    points = find_largest_contour(img)
    points = smooth_contour(points)
    N = 400
    points = resample_contour(points, N)
    points = points.reshape(-1, 2)
    #points[:,1] = img.shape[0] - points[:,1]
    segments = []
    for i in range(N):
        segments.append((i, (i+1)%N))
    segments = np.array(segments)
    A = dict(vertices=points)
    B = tr.triangulate(A, 'q')
    verts,  triangles = remove_outside_triangles(B['vertices'], B['triangles'], img)
    handle_idxs = np.array([10, 30, 50, 100, 200, 300, 400, 500])
    handle_cos =  verts[handle_idxs, :]
    handle_cos = handle_cos + np.random.rand(len(handle_idxs), 2) * 300

    verts_1 = solve_biharmonic(verts, triangles, handle_idxs, handle_cos)
    verts_1[:,1] = img.shape[0] - verts_1[:,1]

    A['vertices'][:,1] = img.shape[0] - A['vertices'][:,1]

    B['vertices'] = verts_1
    B['triangles'] = triangles

    tr.compare(plt, A, B)
    plt.show()

