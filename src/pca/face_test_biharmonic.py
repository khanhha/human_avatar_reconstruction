import os
import sys
import numpy as np
sys.path.insert(0, '/home/khanhhh/data_1/sample_codes/libigl/python')
import pyigl as igl

if __name__ == '__main__':
    print('hello')

    bc_frac = 1.0
    bc_dir = -0.03
    deformation_field = False

    V = igl.eigen.MatrixXd()
    U = igl.eigen.MatrixXd()
    V_bc = igl.eigen.MatrixXd()
    U_bc = igl.eigen.MatrixXd()

    # Z = igl.eigen.MatrixXd()
    F = igl.eigen.MatrixXi()
    b = igl.eigen.MatrixXi()

    dir = '/home/khanhhh/data_1/projects/Oh/data/face/test_biharmonic/'
    obj_path = os.path.join(*[dir, 'vic_head.obj'])
    vmap_path = os.path.join(*[dir, 'vmap_face_head_biharmonic.npy'])
    df_face_path = os.path.join(*[dir, 'vic_deformed_face.obj'])
    out_path = os.path.join(*[dir, 'out_df_head.obj'])
    out_path_1 = os.path.join(*[dir, 'out_df_head_no_biharmonic.obj'])

    igl.readOBJ(obj_path, V, F)
    U = igl.eigen.MatrixXd(V)

    S = igl.eigen.MatrixXd()
    S.resize(V.rows(), 1)
    vmap = np.load(vmap_path)
    for i in range(S.rows()):
        S[i, 0] = -1
    for idx in vmap:
        S[idx, 0]  =1
    S = S.castint()

    b = igl.eigen.MatrixXd([[t[0] for t in [(i, S[i]) for i in range(0, V.rows())] if t[1] >= 0]]).transpose().castint()

    # Boundary conditions directly on deformed positions
    U_bc.resize(b.rows(), V.cols())
    V_bc.resize(b.rows(), V.cols())
    tmp = igl.eigen.MatrixXi()
    igl.readOBJ(df_face_path, U_bc, tmp)

    for bi in range(0, b.rows()):
        V_bc.setRow(bi, V.row(b[bi]))

    D = igl.eigen.MatrixXd()
    D_bc = U_bc - V_bc
    igl.harmonic(V, F, b, D_bc, 2, D)
    U = V + D

    igl.writeOBJ(out_path, U, F)
    U1 = V
    for bi in range(b.rows()):
        U1.setRow(b[bi], U_bc.row(bi))

    igl.writeOBJ(out_path_1, U1, F)

