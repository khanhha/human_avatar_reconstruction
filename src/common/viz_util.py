from mayavi.mlab import *
from mayavi import mlab
import vtk
import tempfile
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread, imsave

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

@static_vars(init=False)
def project_silhouette_mayavi(verts, triangles, measure_contours = None):
    if project_silhouette_mayavi.init == False:
        #disable vtk warning
        errOut = vtk.vtkFileOutputWindow()
        errOut.SetFileName("VTK Error Out.txt")
        vtkStdErrOut = vtk.vtkOutputWindow()
        vtkStdErrOut.SetInstance(errOut)
        project_silhouette_mayavi.init = False

    mlab.options.offscreen = True
    m = triangular_mesh(verts[:, 0], verts[:, 1], verts[:, 2], triangles, color=(0.2,0.4,0.8))

    if measure_contours is not None:
        for name, contour in measure_contours.items():
            mlab.plot3d(contour[:, 0], contour[:, 1], contour[:, 2], tube_radius=0.005, colormap='Spectral')

    with tempfile.TemporaryDirectory() as tmp_dir:
        #front view
        path_f = f'{tmp_dir}/f.png'
        mlab.view(-90, 90)
        mlab.savefig(path_f)
        img = cv.imread(path_f)
        img_f = img[:, 200 - 150:200 + 150, :3]
        img_f = img_f[:,:,::-1]

        #side view
        path_s = f'{tmp_dir}/s.png'
        mlab.view(0, 90)
        mlab.savefig(path_s)
        img = cv.imread(path_s)
        img_s = img[:, 200 - 150:200 + 150, :3]
        img_s = img_s[:,:,::-1]

        #plt.subplot(121); plt.imshow(img_f)
        #plt.subplot(122); plt.imshow(img_s)
        #plt.show()

    #otherwise, it will cause leak mem
    mlab.clf()

    return img_f, img_s

from mpl_toolkits.mplot3d import Axes3D
#TODO: matplotlib suffers from aspect distortion. it makes the mesh look fatter than it actually is
def project_silhouette_matplot(verts, triangles):
    plt.clf()
    ax = plt.gca(projection='3d')
    center = 0.5*(verts.max(0) + verts.min(0))
    verts = verts - center

    # draw the mesh. set fixed color to avoid random color
    ax.plot_trisurf(verts[:,0], verts[:,1], verts[:,2], triangles=triangles, color='grey', linewidth=0, antialiased=False)

    #TODO: trick to force matplotlib to ensure equal scaling across axes
    # Create cubic bounding box to simulate equal aspect ratio
    # link: https://python-decompiler.com/article/2012-12/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
    X, Y, Z = verts[:,0], verts[:,1], verts[:,2]
    xmax, xmin = X.max(), X.min()
    ymax, ymin = Y.max(), Y.min()
    zmax, zmin = Z.max(), Z.min()
    max_range = np.array([xmax- xmin, ymax-ymin, zmax-zmin]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(xmax+xmin)
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(ymax+ymin)
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(zmax+zmin)
    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')

    plt.axis('off')
    plt.grid(b=None)

    #TODO: fixed image range extraction for dpi = 300 to crop the silhouette region in image.
    # if you change dpi to another different number, you have to calibrate this fixed range.
    dpi = 300
    yrange_dpi = (700-350,700+350)
    xrange_dpi = (1000-250,1000+250)
    #trick to get image data from matplotlib: save render outptu to a temp file and then read back
    with tempfile.TemporaryDirectory() as tmp_dir:
        path_f = f'{tmp_dir}/img_front.png'
        ax.view_init(0, -90)
        plt.savefig(path_f, dpi = dpi)
        img_f = imread(path_f, mode='RGB')
        img_f = img_f[yrange_dpi[0]:yrange_dpi[1], xrange_dpi[0]:xrange_dpi[1] ,:]

        path_s = f'{tmp_dir}/img_side.png'
        ax.view_init(0, 0)
        plt.savefig(path_s, dpi=dpi)
        img_s = imread(path_s, mode='RGB')
        img_s = img_s[yrange_dpi[0]:yrange_dpi[1], xrange_dpi[0]:xrange_dpi[1], :]

    return img_f, img_s

def resize_height(img_src, img_target):
    tmax = img_target.shape[0]
    smax = img_src.shape[0]
    scale = tmax/smax
    dsize = (int(img_src.shape[1]*scale), img_target.shape[0])
    img_src_1 = cv.resize(img_src, dsize=dsize)
    return img_src_1

def resize_width(img_src, img_target):
    tmax = img_target.shape[1]
    smax = img_src.shape[1]
    scale = tmax/smax
    dsize = (img_target.shape[1], int(scale*img_src.shape[0]))
    img_src_1 = cv.resize(img_src, dsize=dsize)
    return img_src_1

def build_gt_predict_viz(verts, faces, img_f_org, img_s_org):
    img_f_pred, img_s_pred = project_silhouette_mayavi(verts, faces)
    # print(img_f_pred.shape)
    img_f_org = resize_height(np.array(img_f_org), img_f_pred)
    img_s_org = resize_height(np.array(img_s_org), img_s_pred)
    # print(img_f_org.shape, img_f_pred.shape)
    img_f_pair = np.concatenate((img_f_org, img_f_pred), axis=1)
    img_s_pair = np.concatenate((img_s_org, img_s_pred), axis=1)

    img_s_pair = resize_width(img_s_pair, img_f_pair)
    img_full = np.concatenate((img_f_pair, img_s_pair), axis=0)

    return img_full
