from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2 as cv

from opendr.camera import ProjectPoints
from opendr.renderer import ColoredRenderer
from opendr.lighting import LambertianPointLight
from common.obj_util import export_mesh, import_mesh_obj
import matplotlib.pyplot as plt
from tqdm import tqdm
import tempfile

colors = {
    # colorbline/print/copy safe:
    'light_blue': [0.65098039, 0.74117647, 0.85882353],
    'light_pink': [.9, .7, .7],  # This is used to do no-3d
}

def _create_renderer(w=640,
                     h=480,
                     rt=np.zeros(3),
                     t=np.zeros(3),
                     f=None,
                     c=None,
                     k=None,
                     near=.5,
                     far=10.):

    f = np.array([w, w]) / 2. if f is None else f
    c = np.array([w, h]) / 2. if c is None else c
    k = np.zeros(5) if k is None else k

    rn = ColoredRenderer()

    rn.camera = ProjectPoints(rt=rt, t=t, f=f, c=c, k=k)
    rn.frustum = {'near': near, 'far': far, 'height': h, 'width': w}
    return rn


def _rotateY(points, angle):
    """Rotate the points by a specified angle."""
    ry = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
                   [-np.sin(angle), 0., np.cos(angle)]])
    return np.dot(points, ry)

def simple_renderer(rn,
                    verts,
                    faces,
                    yrot=np.radians(120),
                    color=colors['light_pink']):
    # Rendered model color
    rn.set(v=verts, f=faces, vc=color, bgcolor=np.ones(3))
    albedo = rn.vc

    # Construct Back Light (on back right corner)
    rn.vc = LambertianPointLight(
        f=rn.f,
        v=rn.v,
        num_verts=len(rn.v),
        light_pos=_rotateY(np.array([-200, -100, -100]), yrot),
        vc=albedo,
        light_color=np.array([1, 1, 1]))

    # Construct Left Light
    rn.vc += LambertianPointLight(
        f=rn.f,
        v=rn.v,
        num_verts=len(rn.v),
        light_pos=_rotateY(np.array([800, 10, 300]), yrot),
        vc=albedo,
        light_color=np.array([1, 1, 1]))

    # Construct Right Light
    rn.vc += LambertianPointLight(
        f=rn.f,
        v=rn.v,
        num_verts=len(rn.v),
        light_pos=_rotateY(np.array([-500, 500, 1000]), yrot),
        vc=albedo,
        light_color=np.array([.7, .7, .7]))

    return rn.r

def render_model(verts,
                 faces,
                 w,
                 h,
                 cam,
                 near=0.5,
                 far=25,
                 img=None,
                 do_alpha=False,
                 color_id=None):

    rn = _create_renderer(
        w=w, h=h, near=near, far=far, rt=cam.rt, t=cam.t, f=cam.f, c=cam.c)

    # Uses img as background, otherwise white background.
    if img is not None:
        rn.background_image = img / 255. if img.max() > 1 else img

    if color_id is None:
        color = colors['light_blue']
    else:
        color_list = colors.values()
        color = color_list[color_id % len(color_list)]

    imtmp = simple_renderer(rn, verts, faces, color=color)

    return imtmp


from mpl_toolkits.mplot3d import Axes3D
import scipy.io as io
import plotly.graph_objects as go
from scipy.misc import imread
import PIL.Image as Image
import io
import plotly
import plotly.figure_factory as ff

from numpy import mean


def set_aspect_equal_3d(ax):
    """Fix equal aspect bug for 3D plots."""

    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()

    xmean = mean(xlim)
    ymean = mean(ylim)
    zmean = mean(zlim)

    plot_radius = max([abs(lim - mean_)
                       for lims, mean_ in ((xlim, xmean),
                                           (ylim, ymean),
                                           (zlim, zmean))
                       for lim in lims])

    ax.set_xlim3d([xmean - plot_radius, xmean + plot_radius])
    ax.set_ylim3d([ymean - plot_radius, ymean + plot_radius])
    ax.set_zlim3d([zmean - plot_radius, zmean + plot_radius])

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

import open3d as o3d
def draw_mesh_1(verts, faces):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    #key_to_callback = {}
    #key_to_callback[ord(".")] = capture_image
    #o3d.visualization.draw_geometries_with_key_callbacks([mesh],key_to_callback)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    mesh.rotate((-90,0,0), center=True, type=o3d.geometry.RotationType.AxisAngle)
    vis.add_geometry(mesh)
    view = vis.get_view_control()
    print(view)
    img = vis.capture_screen_float_buffer(do_render=True)
    img = np.asarray(img)
    img = (img*255).astype(np.uint8)
    #print(img)
    #vis.destroy_window()
    plt.imshow(img)
    plt.show()

@static_vars(figure=None)
def draw_mesh(verts, faces):
    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
    if draw_mesh.figure is None:
        print('initialize surf')
        draw_mesh.figure = ff.create_trisurf(x=x, y=y, z=z, simplices=faces, show_colorbar=False)

        draw_mesh.figure.update_layout(scene=dict(
            xaxis=dict(nticks=4, range=[-2, 2], ),
            yaxis=dict(nticks=4, range=[-2, 2], ),
            zaxis=dict(nticks=4, range=[-2, 2], ), ),
            width=700,
            margin=dict(r=20, l=10, b=10, t=10))
    #print('draw')
    fig = draw_mesh.figure
    fig.data[0].update(x=x, y=y, z=z)
    # fig = go.Figure(data=[go.Mesh3d(x=x, y=y, z=z, k=faces[:, 0], j=faces[:, 1], i=faces[:, 2],
    #                                color='grey', opacity=1,
    #                                lighting=dict(ambient=0.4, diffuse=0.8, specular=0, roughness=1.0))])

    name = 'eye = (x:1., y:0.0, z:0.)'
    camera = dict(
        eye=dict(x=1., y=0, z=0.)
    )

    fig.update_layout(scene_camera=camera, title=name)

    # fig.write_image(f"{tmp_dir}/img.png", scale=2)
    # img = cv.imread(f'{tmp_dir}/img.png')
    imgdata = fig.to_image(format='png', scale=2)
    I = Image.open(io.BytesIO(imgdata))
    img = np.array(I)
    # img = img[500-300:500+300, 700-150:700+150, :]
    # img = fig.to_image(format='png', scale=2)
    # del fig
    # plotly.purge()
    plt.clf()
    plt.imshow(img)
    #plt.show()

    name = 'eye = (x:0., y:1.0, z:0.)'
    camera = dict(
        eye=dict(x=0., y=-1, z=0.)
    )
    fig.update_layout(scene_camera=camera, title=name)

    #fig.write_image(f"{tmp_dir}/img.png", scale=2)
    #img = cv.imread(f'{tmp_dir}/img.png')
    #img = img[500 - 300:500 + 300, 700 - 150:700 + 150, :]
    #plt.clf()
    #plt.imshow(img)

if __name__ == '__main__':

    #vic_mesh_path = '/home/khanhhh/data_1/projects/Oh/codes/human_estimation/data/meta_data/align_source_vic_mpii.obj'
    #out_mesh_path = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj/victoria_caesar_obj/CSR0097A.obj'

    vic_mesh_path = '/media/F/projects/Oh/data/cnn/sil_384_256_ml_fml_pose_nosyn_color/log/s/log_mesh/epoch_0/designer_0_front.obj'
    vic_mesh_path_1 = '/media/D1/data_1/projects/Oh/codes/human_estimation/data/meta_data_shared/vic_mesh_only_triangle.obj'
    verts, _ = import_mesh_obj(vic_mesh_path)
    _, faces = import_mesh_obj(vic_mesh_path_1)
    faces = np.array(faces)

    #temp = np.copy(verts[:, 0])
    #verts[:,0] = verts[:,1]
    #verts[:,1] = temp
    #verts[:, 0]  = verts[:,1]
    center = 0.5*(verts.max(0) + verts.min(0))
    verts = verts - center
    #verts = np.swapaxes(verts, 0, 1)

    #x,y,z = verts[:,0], verts[:,1], verts[:,2]
    #fig = ff.create_trisurf(x=x, y=y,z=z, simplices=faces, show_colorbar=False)
    #cnt = 0
    #while True:
    #    print('draw iter ', cnt); cnt +=1
    #    draw_mesh_1(verts, faces)

    vmax = verts.max(0)
    vmin = verts.min(0)
    size = verts.max(0) - verts.min(0)
    smin = size.min()
    smax = 0.5*size.max()
    aspect = (vmax[2]-vmin[2])/(vmax[1]-vmin[1])
    w = 4
    h = aspect * w
    fig = plt.figure(figsize=(w,h))
    #ax = plt.gca(projection='3d')
    ax = Axes3D(fig, proj_type='ortho')
    #ax.set_aspect('equal')
    #ax.set_xlim3d([-smax, smax])
    #ax.set_ylim3d([-smax, smax])
    #ax.set_zlim3d([-0.7*smax, 0.7*smax])
    #ax.set_xlim3d(vmin[0], vmax[0], auto=True)
    #ax.set_ylim3d(vmin[1], vmax[1], auto=True)
    #ax.set_zlim3d(vmin[2], vmax[2], auto=True)
    #ax.axis('scaled')
    #set_axes_equal(ax)
    #ax.auto_scale_xyz([0.0, 1.0], [0.0, 1.0], [0.0, 1.0])
    #ax.autoscale_view()
    #ax.autoscale()

    print(ax.get_proj())
    ax.set_proj_type('ortho')
    ax.plot_trisurf(verts[:,0], verts[:,1], verts[:,2], triangles=faces, linewidth=0, antialiased=False)
    set_aspect_equal_3d(ax)
    #ax.autoscale()
    #ax.autoscale_view()
    plt.axis('off')
    plt.grid(b=None)
    img = None

    # TODO: fixed image range extraction for dpi = 300 to crop the silhouette region in image.
    # if you change dpi to another different number, you have to calibrate this fixed range.
    dpi = 300
    yrange = (700-350,700+350)
    xrange = (1000-250,1000+250)
    #trick to get image data from matplotlib: save render outptu to a temp file and then read back
    with tempfile.TemporaryDirectory() as tmp_dir:
        path_f = f'{tmp_dir}/img_front.png'
        ax.view_init(0, -90)
        plt.show()
        plt.savefig(path_f, dpi = dpi)
        img_f = imread(path_f, mode='RGB')
        img_f = img_f[yrange[0]:yrange[1], xrange[0]:xrange[1] ,:]

        path_s = f'{tmp_dir}/img_side.png'
        ax.view_init(0, 0)
        plt.savefig(path_s, dpi=dpi)
        img_s = imread(path_s, mode='RGB')
        img_s = img_s[yrange[0]:yrange[1], xrange[0]:xrange[1], :]

    img_full = np.concatenate((img_f, img_s), axis=1)
    plt.clf()
    plt.imshow(img_full)
    plt.show()

    bmin = verts.min(0)
    bmax = verts.max(0)
    print(bmin, bmax)
    mheight = (bmax-bmin).max()

    color_id = None
    img = None

    img_size = 512
    w = img_size
    h = img_size
    #flength = mheight*5
    flength = 100
    cam = [flength, w / 2., h / 2.]

    rg = range(0, 100, 2)
    with tqdm(total=len(rg)) as tqdm:
        for i in rg:
            z_cam = 2
            use_cam = ProjectPoints(
                f=cam[0] * np.ones(2),
                rt=np.zeros(3),
                t=np.array((0,5,z_cam)),
                k=np.zeros(5),
                c=cam[1:3])

            #near = np.maximum(np.min(verts[:, 2]) - 10, 0.1)
            #far = np.maximum(np.max(verts[:, 2]) + 10, 10)
            near = 0.1
            far = 15
            far += z_cam
            imtmp = render_model(
                verts,
                faces,
                w,
                h,
                use_cam,
                do_alpha=False,
                img=img,
                far=far,
                near=near,
                color_id=color_id)

            img_ret = (imtmp * 255).astype('uint8')
            plt.clf()
            plt.imshow(img_ret)
            plt.title(f"{i}")
            plt.show()

            tqdm.update(1)