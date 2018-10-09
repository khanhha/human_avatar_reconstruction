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
