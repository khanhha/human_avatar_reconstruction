import numpy as np
import cv2 as cv
from mayavi.mlab import *
from mayavi import mlab
from common.obj_util import export_mesh, import_mesh_obj
import matplotlib.pyplot as plt

def test_mesh():
    """A very pretty picture of spherical harmonics translated from
    the octaviz example."""
    pi = np.pi
    cos = np.cos
    sin = np.sin
    dphi, dtheta = pi / 250.0, pi / 250.0
    [phi, theta] = np.mgrid[0:pi + dphi * 1.5:dphi,
                            0:2 * pi + dtheta * 1.5:dtheta]
    m0 = 4
    m1 = 3
    m2 = 2
    m3 = 3
    m4 = 6
    m5 = 2
    m6 = 6
    m7 = 4
    r = sin(m0 * phi) ** m1 + cos(m2 * phi) ** m3 + \
        sin(m4 * theta) ** m5 + cos(m6 * theta) ** m7
    x = r * sin(phi) * cos(theta)
    y = r * cos(phi)
    z = r * sin(phi) * sin(theta)

    return mesh(x, y, z, colormap="bone")


vic_mesh_path = '/media/F/projects/Oh/data/cnn/sil_384_256_ml_fml_pose_nosyn_color/log/s/log_mesh/epoch_0/designer_0_front.obj'
vic_mesh_path_1 = '/media/D1/data_1/projects/Oh/codes/human_estimation/data/meta_data_shared/vic_mesh_only_triangle.obj'
verts, _ = import_mesh_obj(vic_mesh_path)
_, faces = import_mesh_obj(vic_mesh_path_1)
faces = np.array(faces)
#test_mesh()
#
import vtk
errOut = vtk.vtkFileOutputWindow()
errOut.SetFileName("VTK Error Out.txt")
vtkStdErrOut = vtk.vtkOutputWindow()
vtkStdErrOut.SetInstance(errOut)
while True:
    #mlab.clf()
    mlab.options.offscreen=True
    m = triangular_mesh(verts[:,0], verts[:,1], verts[:,2], faces, color=(0.2,0.4,0.8))
    #mlab.view(0, 90)
    mlab.view(-90, 90)
    mlab.savefig('./test.png')
    img = cv.imread('./test.png')
    img = img[:,200-150:200+150,:]
    plt.imshow(img[:,:,::-1])
    plt.show()
    mlab.view(0, 90)
    mlab.savefig('./test.png')
    img = cv.imread('./test.png')
    img = img[:,200-150:200+150,:]
    plt.imshow(img[:,:,::-1])
    plt.show()
    #del m
    mlab.clf()
    #break
#mlab.show()
