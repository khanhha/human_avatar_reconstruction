import numpy as np
import os
import cv2 as cv
from pathlib import Path

def export_vertices(fpath, obj):
    with open(fpath, 'w') as f:
        for i in range(obj.shape[0]):
            co = tuple(obj[i, :])
            f.write("v %.8f %.8f %.8f \n" % co)

def export_mesh(fpath, verts, faces, add_one = True):
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

def export_mesh_tex_obj(fpath, mesh, add_one = True, img_tex = None):
    assert 'v' in mesh, 'missing v'
    assert 'vt' in mesh , 'missing vt'
    assert 'f' in mesh, 'missing f'
    assert 'ft' in mesh, 'missing ft'
    assert len(mesh['f']) == len(mesh['ft'])

    verts = mesh['v']
    verts_tex = mesh['vt']
    faces = mesh['f']
    faces_tex = mesh['ft']

    dir = Path(fpath).parent
    name = Path(fpath).stem

    mtl_fname = f'{name}.mtl'
    tex_fname = f'{name}.jpg'

    #write texture
    if img_tex is not None:
        cv.imwrite(os.path.join(*[dir, tex_fname]), img_tex)

    #write mtl file
    with open(os.path.join(*[dir, mtl_fname]), 'w') as file:
        file.write(f'map_Kd ./{tex_fname}')

    #write obj file
    with open(fpath, 'w') as f:
        f.write(f'mtllib ./{mtl_fname}\n')
        for i in range(verts.shape[0]):
            co = tuple(verts[i, :])
            f.write("v %.8f %.8f %.8f \n" % co)

        for i in range(verts_tex.shape[0]):
            co = tuple(verts_tex[i, :])
            f.write("vt %.8f %.8f\n" % co)

        for i in range(len(faces)):
            f.write("f")
            for v_idx, v_t_idx in zip(faces[i], faces_tex[i]):
                if add_one == True:
                    v_idx += 1
                    v_t_idx += 1
                f.write(" %d/%d" % (v_idx, v_t_idx))
            f.write("\n")


def import_mesh_obj(fpath):
    coords = []
    faces =  []
    with open(fpath, 'r') as obj:
        file = obj.read()
        lines = file.splitlines()
        for line in lines:
            elem = line.split()
            if elem:
                if elem[0] == 'v':
                    coords.append((float(elem[1]), float(elem[2]), float(elem[3])))
                elif elem[0] == 'vt' or elem[0] == 'vn' or elem[0] == 'vp':
                    #raise Exception('load obj file: un-supported texture, normal...')
                    continue
                elif elem[0] == 'f':
                    f = []
                    for v_idx_str in elem[1:]:
                        v_idx = v_idx_str.split('//')[0]
                        f.append(int(v_idx)-1)
                    faces.append(f)

    return np.array(coords), faces

def import_mesh_tex_obj(fpath):
    verts = []
    verts_tex = []
    faces =  []
    faces_tex =  []
    verts_n = []
    with open(fpath, 'r') as obj:
        file = obj.read()
        lines = file.splitlines()
        for line in lines:
            elem = line.split()
            if elem:
                if elem[0] == 'v':
                    verts.append((float(elem[1]), float(elem[2]), float(elem[3])))
                elif elem[0] == 'vt':
                    verts_tex.append((float(elem[1]), float(elem[2])))
                elif elem[0] == 'vn':
                    verts_n.append((float(elem[1]), float(elem[2]), float(elem[3])))
                elif elem[0] == 'vp':
                    raise Exception('unsupported format')
                elif elem[0] == 'f':
                    f = []
                    ft = []
                    for v_idx_str in elem[1:]:
                        v_idx_str = v_idx_str.replace('//', '/')
                        v_idx = v_idx_str.split('/')
                        if len(v_idx) != 2:
                            raise Exception('unsupported format')
                        f.append(int(v_idx[0])-1)
                        ft.append(int(v_idx[1])-1)
                    faces.append(f)
                    faces_tex.append(ft)
    mesh = {}
    mesh['v']   = np.array(verts)
    mesh['vt']  = np.array(verts_tex)
    mesh['vn'] = np.array(verts_n)
    mesh['f']   = faces
    mesh['ft']  = faces_tex
    return mesh

def load_vertices(fpath):
    coords = []
    with open(fpath, 'r') as obj:
        file = obj.read()
        lines = file.splitlines()
        for line in lines:
            elem = line.split()
            if elem:
                if elem[0] == 'v':
                    coords.append((float(elem[1]), float(elem[2]), float(elem[3])))
    return np.array(coords)


