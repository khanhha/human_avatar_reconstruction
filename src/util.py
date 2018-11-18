import numpy as np

def normalize(vec):
    len = np.linalg.norm(vec)
    if len > 0:
        return vec/len
    else:
        return vec

def rect_plane(rect):
    n = np.cross(rect[2]-rect[0], rect[1]-rect[0])
    n = n / np.linalg.norm(n)
    p = np.mean(rect, axis=0)
    return p,n

def dst_point_plane(point, plane_point, plane_norm):
    return np.dot(plane_norm, point - plane_point)

def calc_triangle_local_basis(verts, tris):
    basis = np.zeros((len(tris),4, 3), dtype=np.float32)
    for i, t in enumerate(tris):
        basis[i, 0, :] = verts[t[0]]
        basis[i, 1, :] = verts[t[1]] - verts[t[0]]
        basis[i, 2, :] = verts[t[2]] - verts[t[0]]
        basis[i, 3, :] = normalize(np.cross(basis[i, 1, :], basis[i, 2, :]))
    return basis

from scipy.ndimage import filters
def smooth_contour(X, Y, sigma=3):
    X_1 = filters.gaussian_filter1d(X, sigma=sigma)
    Y_1 = filters.gaussian_filter1d(Y, sigma=sigma)
    return X_1, Y_1

def isect_line_line(p1, p2, p3, p4):
    a = (p1[0]-p3[0])*(p3[1]-p4[1]) - (p1[1]-p3[1])*(p3[0]-p4[0])
    b = (p1[0]-p2[0])*(p3[1]-p4[1]) - (p1[1]-p2[1])*(p3[0]-p4[0])
    t = a/b
    p = np.array([p1[0]+t*(p2[0]-p1[0]), p1[1]+t*(p2[1]-p1[1])])
    return p

def contour_center(X, Y):
    idx_ymax, idx_ymin = np.argmax(Y), np.argmin(Y)
    center_y = 0.5 * (Y[idx_ymax] + Y[idx_ymin])
    center_x = X[idx_ymin]
    return np.array([center_x, center_y])

def reconstruct_slice_contour(feature, D, W, mirror = False):
    p0 = np.array([D*feature[-2]*feature[-1], 0])
    n_points = len(feature) - 2
    half_idx = int(np.ceil(n_points / 2.0))
    assert half_idx == 4

    prev_p = p0

    points = []
    points.append(prev_p)

    angle_step = np.pi / float(n_points)
    for i in range(1, n_points+1):
        angle = float(i)*angle_step
        #TODO
        # if i == 4:
        #     x, y = 0.0, W/2.0
        # else:
        #     x = (prev_p[1] - feature[i] * prev_p[0]) / (np.tan(angle) - feature[i])
        #     y = np.tan(angle) * x
        #prev_p = np.array([x,y])
        #points.append(prev_p)

        # test
        l0_p0 = prev_p
        l0_p1 = prev_p + np.array([1.0, feature[i-1]])

        l1_p0 = np.array([0.0, 0.0])
        if i == half_idx:
            l1_p1 = np.array([0, W/2])
        elif i < half_idx:
            l1_p1 = np.array([1.0, np.tan(angle)])
        else:
            l1_p1 = np.array([-1.0, -np.tan(angle)])

        isct = isect_line_line(l0_p0, l0_p1, l1_p0, l1_p1)

        points.append(isct)

        prev_p = isct

        #print(isct[0] - x, isct[1] - y)
        #plt.plot([l0_p0[0], l0_p1[0]], [l0_p0[1], l0_p1[1]], 'r-')
        #plt.plot([l1_p0[0], l1_p1[0]], [l1_p0[1], l1_p1[1]], 'b-')
        #plt.plot(isct[0], isct[1], 'b+')

    X = np.array([p[0] for p in points])
    Y = np.array([p[1] for p in points])

    if mirror == True:
        X_mirror = X[1:-1][::-1]
        Y_mirror = -Y[1:-1]

        X = np.concatenate([X,X_mirror], axis=0)
        Y = np.concatenate([Y,Y_mirror], axis=0)

    return np.vstack([X, Y])
