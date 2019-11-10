from mayavi.mlab import *
from mayavi import mlab
import vtk
import tempfile
import cv2 as cv
import numpy as np

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

def measure_color_defs():
    colors = {}
    colors['body'] = 'dimgray'
    colors['m_circ_neck'] = 'blueviolet'
    colors['m_circ_bust'] = 'brown'
    colors['m_circ_underbust'] = 'cadetblue'
    colors['m_circ_upperbust'] = 'chartreuse'
    colors['m_circ_waist'] = 'cornflowerblue'
    colors['m_circ_highhip'] = 'crimson'
    colors['m_circ_hip'] = 'cyan'
    colors['m_circ_thigh'] = 'darkblue'
    colors['m_circ_knee'] = 'darkgoldenrod'

    #we don't have unique contours for these measuremets so far
    colors['m_len_front_body'] = 'darkgray'
    colors['m_len_half_girth'] = 'darkgreen'
    colors['m_len_bikini_girth'] = 'darkkhaki'
    colors['m_len_full_girth'] = 'darkolivegreen'

    colors['m_len_sleeve'] = 'darkred'
    colors['m_len_upperarm'] = 'darkseagreen'

    colors['m_len_waist_knee'] = 'deeppink'
    colors['m_len_skirt_waist_to_hem'] = 'deepskyblue'

    colors['m_circ_upperarm'] = 'goldenrod'
    colors['m_circ_elbow'] = 'indianred'
    colors['m_circ_wrist'] = 'aquamarine'

    return colors

def measurement_colors():
    from matplotlib import colors as pltcolors
    # colors = {}
    # colors['body'] = (89, 89, 89)
    # colors['m_circ_neck'] = (68, 102, 0)
    # colors['m_circ_bust'] = (255, 51, 0)
    # colors['m_circ_underbust'] = (255, 153, 0)
    # colors['m_circ_upperbust'] = (153, 204, 0)
    # colors['m_circ_waist'] = (0, 0, 179)
    # colors['m_circ_highhip'] = (255, 26, 255)
    # colors['m_circ_hip'] = (0, 128, 0)
    # colors['m_circ_thigh'] = (0, 102, 153)
    # colors['m_circ_knee'] = (37, 72, 142)
    #
    # #we don't have unique contours for these measuremets so far
    # colors['m_len_front_body'] = (0, 153, 115)
    # colors['m_len_half_girth'] = (0, 153, 115)
    # colors['m_len_bikini_girth'] = (0, 153, 115)
    # colors['m_len_full_girth'] = (0, 153, 115)
    #
    # colors['m_len_sleeve'] = (102, 26, 255)
    # colors['m_len_upperarm'] = (51, 102, 0)
    #
    # colors['m_len_waist_knee'] = (255, 128, 0)
    # colors['m_len_skirt_waist_to_hem'] = (255, 51, 0)
    #
    # colors['m_circ_upperarm'] = (0, 68, 204)
    # colors['m_circ_elbow'] = (0, 68, 204)
    # colors['m_circ_wrist'] = (0, 68, 204)

    cnames = measure_color_defs()
    colors = dict(map(lambda iter : (iter[0], pltcolors.to_rgba(iter[1])[:3]), cnames.items()))
    return colors

def pick_color(id):
    colors = measurement_colors()
    #255 to 1.0
    default = (0.4, 0.4, 0.4)
    if id in colors.keys():
        return colors[id]
    else:
        return default

def tube_radius(id):
    rad = 0.005
    return rad

@static_vars(init=False)
def project_silhouette_mayavi(verts, triangles, measure_contours = None, ortho_proj = False, body_opacity = 1.0):
    if project_silhouette_mayavi.init == False:
        #disable vtk warning
        errOut = vtk.vtkFileOutputWindow()
        errOut.SetFileName("VTK Error Out.txt")
        vtkStdErrOut = vtk.vtkOutputWindow()
        vtkStdErrOut.SetInstance(errOut)
        project_silhouette_mayavi.init = False

    mlab.options.offscreen = True
    g = 0.8
    mlab.gcf().scene.background = tuple(3*[g])

    m = triangular_mesh(verts[:, 0], verts[:, 1], verts[:, 2], triangles, color=pick_color('body'), opacity=body_opacity)

    if measure_contours is not None:
        for name, contour in measure_contours.items():
            mlab.plot3d(contour[:, 0], contour[:, 1], contour[:, 2], tube_radius=tube_radius(name), color = pick_color(name))

    #must be set after body mesh and contours. dont know why
    if ortho_proj == True:
        mlab.gcf().scene.parallel_projection = True

    figsize = (900, 1200)
    with tempfile.TemporaryDirectory() as tmp_dir:
        #front view
        path_f = f'{tmp_dir}/f.png'
        mlab.view(-90, 90)
        #mlab.show()
        mlab.savefig(path_f, size=figsize)
        img_f = cv.imread(path_f)
        #plt.imshow(img)
        #plt.show()
        #img_f = img_f[:, 200 - 150:200 + 150, :3]
        img_f = img_f[:,:,::-1]

        #side view
        path_s = f'{tmp_dir}/s.png'
        mlab.view(0, 90)
        mlab.savefig(path_s, size=figsize)
        img_s = cv.imread(path_s)
        #img_s = img_s[:, 200 - 150:200 + 150, :3]
        img_s = img_s[:,:,::-1]

        #plt.subplot(121); plt.imshow(img_f)
        #plt.subplot(122); plt.imshow(img_s)
        #plt.show()

    #otherwise, it will cause leak mem
    mlab.clf()

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

def build_gt_predict_viz(verts, faces, img_f_org, img_s_org, measure_contours = None, ortho_proj = False, body_opacity = 1.0):
    img_f_pred, img_s_pred = project_silhouette_mayavi(verts, faces, measure_contours, ortho_proj=ortho_proj, body_opacity=body_opacity)
    # print(img_f_pred.shape)
    img_f_org = resize_height(np.array(img_f_org), img_f_pred)
    img_s_org = resize_height(np.array(img_s_org), img_s_pred)
    # print(img_f_org.shape, img_f_pred.shape)
    img_f_pair = np.concatenate((img_f_org, img_f_pred), axis=1)
    img_s_pair = np.concatenate((img_s_org, img_s_pred), axis=1)

    img_s_pair = resize_width(img_s_pair, img_f_pair)
    img_full = np.concatenate((img_f_pair, img_s_pair), axis=0)

    return img_full

def gen_measurement_color_annotation():
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    patches = []
    for key, color in measurement_colors().items():
        red_patch = mpatches.Patch(color=color, label=key)
        patches.append(red_patch)
    plt.legend(handles=patches)
    plt.show()
