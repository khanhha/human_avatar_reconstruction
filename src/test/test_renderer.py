from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2

from opendr.camera import ProjectPoints
from opendr.renderer import ColoredRenderer
from opendr.lighting import LambertianPointLight
from common.obj_util import export_mesh, import_mesh_obj
import matplotlib.pyplot as plt
from tqdm import tqdm
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


if __name__ == '__main__':

    vic_mesh_path = '/home/khanhhh/data_1/projects/Oh/codes/human_estimation/data/meta_data/align_source_vic_mpii.obj'
    out_mesh_path = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj/victoria_caesar_obj/CSR0097A.obj'

    verts, faces = import_mesh_obj(vic_mesh_path)
    verts *= 0.05
    faces = np.array(faces)

    temp = np.copy(verts[:, 1])
    verts[:,1] = verts[:,2]
    verts[:,2] = temp

    verts[:, 1]  = -verts[:,1]
    #verts[:,2]  += 50



    color_id = None
    img = None

    img_size = 512
    w = img_size
    h = img_size
    flength = 500

    cam = [flength, w / 2., h / 2.]

    rg = range(0, 100, 2)
    with tqdm(total=len(rg)) as tqdm:
        for i in rg:
            z_cam = 100
            use_cam = ProjectPoints(
                f=cam[0] * np.ones(2),
                rt=np.zeros(3),
                t=np.array((0,60,z_cam)),
                k=np.zeros(5),
                c=cam[1:3])

            near = np.maximum(np.min(verts[:, 2]) - 25, 0.1)
            far = np.maximum(np.max(verts[:, 2]) + 25, 25)
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