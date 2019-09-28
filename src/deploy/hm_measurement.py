import numpy as np
import pickle
from common.obj_util import import_mesh_obj
from pathlib import Path
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from common.transformations import *
import argparse
from common.obj_util import import_mesh_tex_obj

def fit_plane(points):
    """
    p, n = planeFit(points)

    Given an array, points, of shape (d,...)
    representing points in d-dimensional space,
    fit an d-dimensional plane to the points.
    Return a point, p, on the plane (the point-cloud centroid),
    and the normal, n.
    """
    import numpy as np
    from numpy.linalg import svd
    points = np.reshape(points, (np.shape(points)[0], -1))  # Collapse trialing dimensions
    assert points.shape[0] <= points.shape[1], "There are only {} points in {} dimensions.".format(points.shape[1],
                                                                                                   points.shape[0])
    ctr = points.mean(axis=1)
    x = points - ctr[:, np.newaxis]
    M = np.dot(x, x.T)  # Could also use np.cov(x) here.
    return ctr, svd(M)[0][:, -1]

def center_bb(verts):
    x = 0.5 * (verts[:, 0].max() + verts[:, 0].min())
    y = 0.5 * (verts[:, 1].max() + verts[:, 1].min())
    z = 0.5 * (verts[:, 2].max() + verts[:, 2].min())
    return np.array([x, y, z])

def calc_convex_circumference(verts):
    from shapely.geometry import MultiPoint
    sh_points = MultiPoint(verts[:, :2])
    sh_convex = sh_points.convex_hull
    return sh_convex.length

def align_vertical_axis(points):
    plane_p, plane_n = fit_plane(points.T)
    z = np.array([0.0, 0.0, 1.0])
    M = rotation_matrix(angle_between_vectors(z, plane_n), vector_product(z, plane_n))

    center = center_bb(points)
    points = np.dot(points - center, M[:3, :3]) + center
    return points

def calc_circumference(points):
    points = np.vstack([points, points[0, :]])
    segs = np.diff(points, axis=0)
    len_seg = np.linalg.norm(segs, axis=1)
    circ = np.sum(len_seg)
    return circ

def calc_neighbor_idxs(vert_grps, mesh_path):
    from shapely.geometry import LineString, Point
    #mesh_path = '/home/khanhhh/data_1/projects/Oh/codes/human_estimation/data/meta_data/victoria_template.obj'

    z = np.array([0.0, 0.0, 1.0])

    grp_neighbor_idxs = {}

    for grp_name, grp_idxs in vert_grps.items():

        if 'verts_circ_' not in grp_name:
            continue

        N = 40

        verts, faces = import_mesh_obj(mesh_path)

        bust_verts = verts[grp_idxs]
        center = center_bb(bust_verts)
        radius = calc_convex_circumference(bust_verts)/N

        plane_p, plane_n = fit_plane(bust_verts.T)
        M = rotation_matrix(angle_between_vectors(z, plane_n), vector_product(z, plane_n))
        bust_verts = np.dot(bust_verts - center, M[:3,:3]) + center

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.set_aspect(1.0)
        # ax.scatter(bust_verts[:, 0], bust_verts[:, 1], bust_verts[:, 2], 'g+')
        # ax.scatter(bust_verts_1[:, 0], bust_verts_1[:, 1], bust_verts_1[:, 2], 'r+')
        # ax.scatter(center[0], center[1], center[2], 'r+')
        # ax.quiver(center[0], center[1], center[2], z[0], z[1],z[2])
        # ax.quiver(center[0], center[1], center[2], plane_n[0], plane_n[1],plane_n[2])
        # plt.show()

        max_range = bust_verts[:,:2].max() - bust_verts[:,:2].min()

        angle_step = 2*np.pi / N
        points = []
        neighbor_idxs = []
        for i in range(N):
            angle = i * angle_step
            dir = 0.55*max_range*np.array([np.cos(angle), np.sin(angle)])
            sh_dir = LineString([center[:2], center[:2] + dir])
            dsts = []
            for idx in range(bust_verts.shape[0]):
                sh_p = Point(bust_verts[idx, :2])
                dsts.append(sh_dir.distance(sh_p))
            dsts = np.array(dsts)

            cls_idxs = np.argwhere(dsts < radius).flatten()
            neighbor_idxs.append(grp_idxs[cls_idxs])

            #p = np.average(bust_verts[cls_idxs], axis=0, weights=1.0/dsts[cls_idxs])
            p = np.average(bust_verts[cls_idxs], axis=0)
            points.append(p)

            # fig = plt.figure()
            # ax = fig.add_subplot(111)
            # ax.set_aspect(1.0)
            # ax.plot(bust_verts[:, 0], bust_verts[:, 1], 'g+')
            # ax.plot(bust_verts[cls_idxs, 0], bust_verts[cls_idxs, 1], 'y+')
            # #ax.plot(points[:, 0], points[:, 1], 'b+')
            # ax.plot(center[0], center[1], 'r+')
            # plt.show()

        grp_neighbor_idxs[grp_name] = neighbor_idxs

        # points = np.array(points)
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.set_aspect(1.0)
        # ax.plot(bust_verts[:, 0], bust_verts[:, 1], 'g+')
        # ax.plot(points[:, 0], points[:, 1], 'b+')
        # ax.plot(center[0], center[1], 'r+')
        # plt.show()

    return grp_neighbor_idxs

class HumanMeasure():

    def __init__(self, vert_grp_path, contour_circ_neighbor_idxs_path):
        """

        :param vert_grp_path: vertex groups on Victoria's topology to calcalculate measurement. there are two main types of vetex group, which are distinguished by their name
            type_1) measure_circ_* : this type of vertex group refers to a point-cloud ring of vertices around some type of circumference like bust, waist, etc.
            type_2) measure_ld_*: this type of vertex group refers to a set of vertices around landmark locations such as shoulder, joint. the final landmark point will be an avearge of these vertices
        :param contour_circ_neighbor_idxs_path:
            this refer to a pre-calculated filtering structure, based on  measure_circ_*. we know that measure_circ_* refers to a point cloud around measurement circumfernece, we need some way
            to reduce this point-cloud into a contour with connected vertices. this is what contour_circ_neighbor_idxs_path used for. each of its element refer to a set of neighboring vertex indices
            which will be averaged to calculate the corresponding contour vertex.
        """
        with open(vert_grp_path, 'rb') as file:
            self.vert_grps = pickle.load(file=file)

        with open(contour_circ_neighbor_idxs_path, 'rb') as file:
            self.contour_circ_neighbor_idxs = pickle.load(file=file)

    def measure(self, verts):
        """
        :param verts: customer vertices in form of Victoria's topology
        :return: a dict of measurement values
        - m_circ_* items refer to circumference values like bust, waist, etc.
        - m_len_* items refer to length values like upper arm, shoulder-to-wasit, etc.
        """
        assert verts.shape[0] == 49963, 'unexpected number of vertices'
        assert verts.shape[1] == 3, 'unexpected vertex format'

        circ_contours = self._filter_circumference_contours(verts)

        landmarks = self.filter_landmarks(self.vert_grps, verts)

        measure = self.calc_measurements(circ_contours, landmarks)

        maxv = verts.max(0)
        minv = verts.min(0)
        dif = maxv - minv
        h = dif.max()
        measure['height'] = h

        return measure

    def _filter_circumference_contours(self, verts):
        """
        reduce the point-cloud vertices of circumfernces to closed contours
        :param filter_neighbor_idxs:
        :param verts:
        :return:
        """
        filtered_contours = dict()

        for grp_name, neighbor_idxs in self.contour_circ_neighbor_idxs.items():
            points = []

            #averge a set of neighboring vertices to a contour point
            for idxs in neighbor_idxs:
                p = np.mean(verts[idxs, :], axis=0)
                points.append(p)

            points = np.array(points)
            points = align_vertical_axis(points)
            filtered_contours[grp_name] = points

            # if 'upperarm' in grp_name:
            #     fig = plt.figure()
            #     ax = fig.add_subplot(111)
            #     ax.set_aspect(1.0)
            #     ax.plot(points[:,0], points[:,1], 'r-')
            #     ax.set_title(f'{path.name}-{grp_name}')
            #     plt.show()
            #
            #     fig = plt.figure()
            #     ax = fig.add_subplot(111, projection='3d')
            #     ax.set_aspect(1.0)
            #     ax.scatter(points[:, 0], points[:, 1], points[:, 2], 'r+')
            #     ax.scatter(all_points[:, 0], all_points[:, 1], all_points[:, 2], 'r+')
            #     ax.set_title(path.name)
            #     plt.show()

        return filtered_contours


    def filter_landmarks(self, vert_group_idxs, verts):
        """
        average point cloud around each landmark to a single landmark point.
        this is necessary to reduce the noise because it's not always ideal that a vertex at index K refer to shoulder landamrk.
        we need a set of vertices around K to estiate shoulder landmark
        :param vert_group_idxs:
        :param verts:
        :return:
        """
        ld_points = dict()
        for grp_name, idxs in vert_group_idxs.items():
            if 'verts_ld' not in grp_name:
                continue
            avg = np.mean(verts[idxs, :], axis=0)
            ld_points[grp_name] = avg

        return ld_points

    def calc_measurements(self, contours, landmarks):
        M = dict()

        M['m_circ_neck'] = calc_circumference(contours['verts_circ_neck'])
        M['m_circ_bust'] = calc_circumference(contours['verts_circ_bust'])
        M['m_circ_underbust'] = calc_circumference(contours['verts_circ_underbust'])
        M['m_circ_waist'] = calc_circumference(contours['verts_circ_waist'])
        M['m_circ_midwaist'] = calc_circumference(contours['verts_circ_midwaist'])
        M['m_circ_thigh'] = calc_circumference(contours['verts_circ_thigh'])
        M['m_circ_hip'] = calc_circumference(contours['verts_circ_hip'])
        M['m_circ_knee'] = calc_circumference(contours['verts_circ_knee'])

        M['m_circ_upperarm'] = calc_circumference(contours['verts_circ_upperarm'])
        M['m_circ_elbow'] = calc_circumference(contours['verts_circ_elbow'])
        M['m_circ_wrist'] = calc_circumference(contours['verts_circ_wrist'])

        under_neck = landmarks['verts_ld_shoulder']
        elbow = np.mean(contours['verts_circ_elbow'], axis=0)
        M['m_len_front_bodice'] = abs(under_neck[2] - elbow[2])

        shoulder = landmarks['verts_ld_underneck']
        M['m_len_upperarm'] = np.linalg.norm(shoulder - elbow)

        knee = np.mean(contours['verts_circ_knee'], axis=0)
        waist = np.mean(contours['verts_circ_waist'], axis=0)
        M['m_len_waist_knee'] = np.linalg.norm(waist - knee)

        wrist = np.mean(contours['verts_circ_wrist'], axis=0)
        M['m_len_sleeve'] = np.linalg.norm(wrist - shoulder)

        hem = landmarks['verts_ld_hem']
        M['m_len_skirt_waist_to_hem'] = abs(waist[2] - hem[2])

        return M

def calculate_contour_circ_neighbor_idxs(out_dir, measure_vert_grps_path, mesh_path):
    """
    this function just need to be called once to pre-calculate neighboring structure for each point-cloud circumference
    """
    with open(measure_vert_grps_path, 'rb') as file:
        vert_grps = pickle.load(file=file)

    contour_circ_neighbor_idxs =  calc_neighbor_idxs(vert_grps, mesh_path)
    out_path = f'{out_dir}/victoria_measure_contour_circ_neighbor_idxs.pkl'
    with open(out_path, 'wb') as file:
        pickle.dump(obj=contour_circ_neighbor_idxs, file=file)

class HmJointEstimator():
    def __init__(self, joint_vertex_groups_path):
        with open(joint_vertex_groups_path, 'rb') as file:
            self.joint_vert_groups = pickle.load(file)
            #print(self.joint_vert_groups)

    def estimate_joints(self, mesh_verts):
        joints = {}
        for joint_name, joint_v_idxs in self.joint_vert_groups.items():
            joint_verts = mesh_verts[joint_v_idxs, :]
            joint = np.mean(joint_verts, 0)
            joints[joint_name] = joint
        return joints

def main_test():
    ap = argparse.ArgumentParser()
    ap.add_argument("-obj", type=str, required=False, help='path to an obj file of victoria topo to measure')
    ap.add_argument("-grp", type=str, required=True, help='meta_data/victoria_measure_vert_groups.pkl')
    ap.add_argument("-nbr", type=str, required=True, help='meta_data/victoria_measure_contour_circ_neighbor_idxs.pkl')
    args = ap.parse_args()

    #measure_vert_grps_path  = '/home/khanhhh/data_1/projects/Oh/codes/human_estimation/data/meta_data/victoria_measure_vert_groups.pkl'
    #circ_neighbor_idxs_path = '/home/khanhhh/data_1/projects/Oh/codes/human_estimation/data/meta_data/victoria_measure_contour_circ_neighbor_idxs.pkl'
    measure_vert_grps_path = args.grp
    circ_neighbor_idxs_path = args.nbr
    assert Path(args.obj).exists(), f'non existing file {args.obj}'
    assert Path(measure_vert_grps_path).exists(),  f'non existing file {measure_vert_grps_path}'
    assert Path(circ_neighbor_idxs_path).exists(), f'non existing file {circ_neighbor_idxs_path}'

    bd_measure = HumanMeasure(vert_grp_path=measure_vert_grps_path, contour_circ_neighbor_idxs_path=circ_neighbor_idxs_path)

    verts, _  = import_mesh_obj(args.obj)
    M = bd_measure.measure(verts)

    print('\n\n')
    print(f'measurement of file {Path(args.obj).name}:\n')
    for name, m in M.items():
        print(name, ": ", m)


    # mesh_dir = '/home/khanhhh/data_1/projects/Oh/data/3d_human/caesar_obj/vic_pca_models_1/pca_coords/debug_female'
    # for path in Path(mesh_dir).glob('*.obj'):
    #
    #     verts, _  = import_mesh(path)
    #
    #     M = bd_measure.measure(verts)
    #
    #     print('\n\n')
    #     print(path.name)
    #     for name, m in M.items():
    #         print(name, ": ", m)

if __name__ == '__main__':
    #main_test()
    dir = '/media/D1/data_1/projects/Oh/codes/human_estimation/data/meta_data_shared/'
    measure_vert_groups_path = f'{dir}/victoria_measure_vert_groups.pkl'
    mesh_path = f'{dir}/predict_sample_mesh.obj'
    calculate_contour_circ_neighbor_idxs(dir, measure_vert_groups_path, mesh_path)