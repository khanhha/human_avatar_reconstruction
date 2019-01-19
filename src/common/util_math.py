import numpy as np
import numpy.linalg as linalg

def normalize(vec):
    len = linalg.norm(vec)
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

def rect_plane(rect):
    n = np.cross(rect[2]-rect[0], rect[1]-rect[0])
    n = n / linalg.norm(n)
    p = np.mean(rect, axis=0)
    return p,n

def dst_point_plane(point, plane_point, plane_norm):
    return np.dot(plane_norm, point - plane_point)

def dst_point_line(point, line_p, line_dir):
    v = point-line_p
    n = np.array([-line_dir[1], line_dir[0]])
    return np.dot(v, n)

def isect_line_line(p1, p2, p3, p4):
    a = (p1[0]-p3[0])*(p3[1]-p4[1]) - (p1[1]-p3[1])*(p3[0]-p4[0])
    b = (p1[0]-p2[0])*(p3[1]-p4[1]) - (p1[1]-p2[1])*(p3[0]-p4[0])
    t = a/b
    p = np.array([p1[0]+t*(p2[0]-p1[0]), p1[1]+t*(p2[1]-p1[1])])
    return p

def closest_on_quad_to_point_v3(p, a, b, c, d):
    p_0 = closest_on_tri_to_point_v3(p, a, b, c)
    p_1 = closest_on_tri_to_point_v3(p, a, c, d)
    if linalg.norm(p - p_0) <  linalg.norm(p - p_1):
        return p_0
    else:
        return p_1

# * Set 'r' to the point in triangle (a, b, c) closest to point 'p' */
def closest_on_tri_to_point_v3(p, a, b, c):
    #Check if P in vertex region outside A
    ab = b-a
    ac = c-a
    ap = p-a
    d1 = np.dot(ab, ap)
    d2 = np.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return a

    # Check if P in vertex region outside B
    bp = p-b
    d3 = np.dot(ab, bp)
    d4 = np.dot(ac, bp)
    if d3 > 0.0 and d4 <= d3:
        return b

    #check if P in edge region of AB, if so return projection of P onto AB
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        #barycentric coordinates (1-v,v,0)
        return a + v * ab

    #Check if P in vertex region outside C
    cp = p-c
    d5 = np.dot(ab, cp)
    d6 = np.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        # barycentric coordinates (0,0,1)
        return c

    #Check if P in edge region of AC, if so return projection of P onto AC
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        #barycentric coordinates (1-w,0,w)
        return a + w * ac

    #Check if P in edge region of BC, if so return projection of P onto BC
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        #barycentric coordinates (0,1-w,w)
        return  b + w*(c - b)

    # P inside face region. Compute Q through its barycentric coordinates (u,v,w)
    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    #  a + ab * v + ac * w
    return a + ab * v + ac * w
