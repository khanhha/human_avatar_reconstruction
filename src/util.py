import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LinearRing, LineString, Point, Polygon, MultiPoint
from shapely.ops import nearest_points
import shapely.affinity as affinity
import math
from numpy.linalg import norm

def normalize(vec):
    len = np.linalg.norm(vec)
    if len > 0:
        return vec/len
    else:
        return vec

#line dir must be normalized
def mirror_point_through_axis(line_p, line_dir, p):
    v = p-line_p
    a = np.dot(v, line_dir)
    proj = line_p + a*line_dir
    return proj + (proj-p)

def furthest_points_line(points ,line_p, line_dir):
    dsts = np.zeros(shape=(points.shape[0],), dtype=np.float)
    for i in range(points.shape[0]):
        dsts[i] = abs(dst_point_line(points[i,:], line_p, line_dir))
    return np.argmax(dsts)

def closest_point_points(point, points):
    g0 = Point(point)
    if len(points) > 1:
        g1 = MultiPoint(points)
        closest = nearest_points(g0, g1)[1]
        return np.array([closest.x, closest.y])
    else:
        return np.array([g0.x, g0.y])

def rect_plane(rect):
    n = np.cross(rect[2]-rect[0], rect[1]-rect[0])
    n = n / np.linalg.norm(n)
    p = np.mean(rect, axis=0)
    return p,n

def dst_point_plane(point, plane_point, plane_norm):
    return np.dot(plane_norm, point - plane_point)

def dst_point_line(point, line_p, line_dir):
    v = point-line_p
    n = np.array([-line_dir[1], line_dir[0]])
    return np.dot(v, n)

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

def is_leg_contour(name):
    if name == 'UnderCrotch':
        return True
    else:
        return False

def contour_center(X, Y):
    idx_ymax, idx_ymin = np.argmax(Y), np.argmin(Y)
    center_y = 0.5 * (Y[idx_ymax] + Y[idx_ymin])
    center_x = X[idx_ymin]
    return np.array([center_x, center_y])

from scipy.interpolate import splprep, splev
def resample_contour(X, Y, n_point):
    tck, u = splprep([X, Y], s=0)
    u_1 = np.linspace(0.0, 1.0, n_point)
    X, Y = splev(u_1, tck)
    return X, Y

def reconstruct_leg_slice_contour(feature, D, W):
    p0 = np.array([D*feature[-2]*feature[-1], 0])
    n_points = len(feature) -1
    prev_p = p0
    points = []
    points.append(prev_p)

    angle_step = 2.0*np.pi / float(n_points)
    for i in range(1, n_points):
        angle = float(i)*angle_step

        l0_p0 = prev_p
        l0_p1 = prev_p + np.array([1.0, feature[i-1]])

        l1_p0 = np.array([0.0, 0.0])
        l1_p1 = np.array([np.cos(angle), np.sin(angle)])

        isct = isect_line_line(l0_p0, l0_p1, l1_p0, l1_p1)

        points.append(isct)

        prev_p = isct

    X = np.array([p[0] for p in points])
    Y = np.array([p[1] for p in points])

    return np.vstack([X, Y])

def reconstruct_torso_slice_contour(feature, D, W, mirror = False):
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
        Y_mirror = -Y[1:-1][::-1]

        X = np.concatenate([X,X_mirror], axis=0)
        Y = np.concatenate([Y,Y_mirror], axis=0)

    return np.vstack([X, Y])

def align_torso_contour(X, Y, anchor_pos_x = True, debug_path = None):
    idx_ymax, idx_ymin = np.argmax(Y), np.argmin(Y)
    center_y = 0.5 * (Y[idx_ymax] + Y[idx_ymin])
    center_x = X[idx_ymin]

    if debug_path is not None:
        plt.clf()
        plt.axes().set_aspect(1)
        plt.plot(X, Y, 'b+')
        plt.plot(center_x, center_y, 'r+')

    contour = Polygon([(x,y) for x, y in zip(X,Y)])
    contour_1 = contour.simplify(0.01, preserve_topology=False)
    contour_2 = contour_1.convex_hull
    for p in contour_2.exterior.coords:
        plt.plot(p[0], p[1], 'r+', ms=7)

    #find the anchor segment
    n_point = len(contour_2.exterior.coords)
    anchor_p1 = None
    anchor_p2 = None
    for i in range(n_point):
        p0 = np.array(contour_2.exterior.coords[i])
        p1 = np.array(contour_2.exterior.coords[(i+1)%n_point])
        c = 0.5*(p0+p1)
        ymin = min(p0[1], p1[1])
        ymax = max(p0[1], p1[1])
        if (anchor_pos_x == True and c[0] > center_x) or (anchor_pos_x == False and c[0] < center_x):
            if  ymin <= center_y and center_y <= ymax:
                anchor_p1 = p0
                anchor_p2 = p1
                if anchor_p1[1] > anchor_p2[1]:
                    anchor_p1, anchor_p2 = anchor_p2, anchor_p1

    if anchor_p1 is None or anchor_p2 is None:
        return None, None

    dir_1 = anchor_p2 - anchor_p1
    anchor_dir = dir_1 / norm(dir_1)
    anchor_dir[1] = abs(anchor_dir[1])

    #rotate the contour to align the anchor line
    angle = math.acos(np.dot(anchor_dir, np.array([0, 1])))
    if anchor_dir[0] < 0:
        angle = -angle

    contour_aligned = affinity.rotate(contour, angle = angle, origin=Point(anchor_p1), use_radians=True)
    X_algn = [p[0] for p in contour_aligned.exterior.coords]
    Y_algn = [p[1] for p in contour_aligned.exterior.coords]

    if debug_path is not None:
        plt.plot(anchor_p1[0], anchor_p1[1], 'r+', ms=14)
        plt.plot(anchor_p2[0], anchor_p2[1], 'r+', ms=14)
        plt.plot([anchor_p1[0], anchor_p2[0]], [anchor_p1[1], anchor_p2[1]], 'r-', ms=14)

        plt.plot(X_algn, Y_algn, 'r-')
        plt.savefig(debug_path)
        #plt.show()

    return np.array(X_algn), np.array(Y_algn)
