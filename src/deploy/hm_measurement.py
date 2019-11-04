import numpy as np
import pickle
from common.obj_util import import_mesh_obj
from pathlib import Path
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from common.transformations import *
import argparse
from common.obj_util import import_mesh_tex_obj
import alphashape
from shapely.geometry.polygon import orient
from shapely.geometry import MultiPolygon, Polygon, MultiPoint
import copy
from common.viz_util import project_silhouette_mayavi, gen_measurement_color_annotation

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


#see picture for more detail
#assets/images/bust_proj.png
def extract_rbreast_cross_contour(contour_points):
    idxs_hor_sorted = np.argsort(contour_points[:, 0])
    #the left most point
    lm_idx = idxs_hor_sorted[0]
    #the right most point
    rm_idxs = idxs_hor_sorted[-5:]
    #for the right most point, we pick the point with lowest y value.
    rm_ys = contour_points[rm_idxs, 1]
    rm_idx = rm_idxs[np.argmin(rm_ys)]
    #rm_idx = rm_idxs[0] if contour_points[rm_idxs[0], 1] < contour_points[rm_idxs[1], 1] else rm_idxs[1]

    if lm_idx < rm_idx:
        verts = contour_points[lm_idx:rm_idx+1, :]
    elif lm_idx > rm_idx:
        verts =  np.vstack([contour_points[lm_idx:, :], contour_points[:rm_idx+1, :]])
    else:
        print('something wrong')
        #fallback to a stupid solution: full contour
        verts = contour_points
    return verts

def calc_cross_right_breast(rbreast_points):
    points = rbreast_points[:,:2]
    ashape = alphashape.alphashape(points, 20)
    ashape = orient(ashape, sign=1)
    contour_points = np.array(ashape.exterior.coords)
    rbreast_contour_points = extract_rbreast_cross_contour(contour_points)
    #plt.clf()
    #plt.axes().set_aspect(1.0)
    #plt.scatter(points[:,0], points[:,1])
    #plt.plot(rbreast_contour_points[:,0], rbreast_contour_points[:,1], '-r', markerSize=4)
    #plt.show()
    diff = np.diff(rbreast_contour_points, axis=0)
    length = np.sum(np.linalg.norm(diff, axis=1))
    z = np.mean(rbreast_points[:,2])
    z_vals = np.full((contour_points.shape[0],1), z)
    contour = np.hstack([contour_points, z_vals])
    return (length, contour)

def extract_breast_depth_contour(contour_points):
    idxs_ver_sorted = np.argsort(contour_points[:, 1])
    #highest point
    start_idx = idxs_ver_sorted[-1]
    #lowest point
    end_idx = idxs_ver_sorted[0]

    if start_idx < end_idx:
        verts = contour_points[start_idx:end_idx+1, :]
    elif start_idx > end_idx:
        verts =  np.vstack([contour_points[start_idx:, :], contour_points[:end_idx+1, :]])
    else:
        print('something wrong')
        #fallback to a stupid solution: full contour
        verts = contour_points

    #return verts

    #find points with highest depth
    n = verts.shape[0]
    depths = verts[-1, 0] - verts[:,0]
    idx = np.argmax(depths)
    if idx >= (n-2):
        #oh fuck. the point with higest depth is very close to the lower point => must be a very very skinny person with flatten bust
        idx = int(0.5*contour_points.shape[0])

    #ok, extract the depth vertical string
    verts = verts[idx:, :]
    return verts


#see picture for more detail
#assets/images/bust_depth.png
def calc_breast_depth(breast_points):
    points = breast_points[:,1:3]
    ashape = alphashape.alphashape(points, 20)
    ashape = orient(ashape, sign=1)
    contour_points = np.array(ashape.exterior.coords)
    breast_contour_points = extract_breast_depth_contour(contour_points)
    #plt.clf()
    #plt.axes().set_aspect(1.0)
    #plt.scatter(points[:,0], points[:,1])
    #plt.plot(breast_contour_points[:,0], breast_contour_points[:,1], '-r', markerSize=4)
    #plt.show()
    diff = np.diff(breast_contour_points, axis=0)
    length = np.sum(np.linalg.norm(diff, axis=1))
    contour_x = np.mean(breast_points[:,0])
    x_vals = np.full((breast_contour_points.shape[0], 1), contour_x)
    contour = np.hstack([x_vals, breast_contour_points])
    return (length, contour)

def convex_contour_points(points):
    pointset_obj = MultiPoint(points)
    convex = pointset_obj.convex_hull
    contour_points = np.array(convex.exterior.coords)
    return contour_points

def calc_point_set_circumference(points, convex_hull = True, alpha_threshold = 5, proj_fit_plane = False):
    trans_mat = None
    if proj_fit_plane:
        N = points.shape[0]
        pln_pnt, pln_n = fit_plane(points.T)

        #proj_mat = projection_matrix(pln_pnt, pln_n)
        #points = np.hstack([points, np.ones((N,1), dtype=points.dtype)])
        #points = np.matmul(proj_mat, points.T).T[:,:3]

        v0 = pln_n
        z = np.array([0.0,0.0, 1.0])
        trans_mat = rotation_matrix(angle_between_vectors(v0, z), vector_product(v0, z), pln_pnt)
        points = np.hstack([points, np.ones((N,1), dtype=points.dtype)])
        points = np.matmul(trans_mat, points.T).T[:,:3]

    point2ds = points[:,:2]
    if convex_hull:
        contour_points = convex_contour_points(point2ds)
    else:
        ashape = alphashape.alphashape(points, alpha_threshold)
        if isinstance(ashape, MultiPolygon):
            # alpha is too big.
            # resort to convex hull
            contour_points = convex_contour_points(point2ds)
        else:
            contour_points = np.array(ashape.exterior.coords)

    #plt.clf()
    #plt.axes().set_aspect(1.0)
    #plt.plot(points[:, 0], points[:, 1], '+r')
    #plt.plot(contour_points[:, 0], contour_points[:, 1], '-b')
    #plt.show()

    dif = np.diff(contour_points, axis=0)
    circ = np.linalg.norm(dif, axis=1).sum()

    contour_z = 0.5 * (points[:, 2].max() + points[:, 2].min())
    z_vals = np.full((contour_points.shape[0], 1), contour_z)

    contour_points = np.hstack([contour_points, z_vals])

    if trans_mat is not None:
        inv = inverse_matrix(trans_mat)
        n = contour_points.shape[0]
        contour_points = np.hstack([contour_points, np.ones((n, 1), dtype=points.dtype)])
        contour_points = np.matmul(inv, contour_points.T).T[:, :3]

    return (circ, contour_points)

class HumanMeasure():

    def __init__(self, vert_grp_path, template_mesh_path):
        """

        :param vert_grp_path: vertex groups on Victoria's topology to calcalculate measurement. there are two main types of vetex group, which are distinguished by their name
            type_1) measure_circ_* : this type of vertex group refers to a point-cloud ring of vertices around some type of circumference like bust, waist, etc.
            type_2) measure_ld_*: this type of vertex group refers to a set of vertices around landmark locations such as shoulder, joint. the final landmark point will be an avearge of these vertices
        :param contour_circ_neighbor_idxs_path:
            this refer to a pre-calculated filtering structure, based on  measure_circ_*. we know that measure_circ_* refers to a point cloud around measurement circumfernece, we need some way
            to reduce this point-cloud into a contour with connected vertices. this is what contour_circ_neighbor_idxs_path used for. each of its element refer to a set of neighboring vertex indices
            which will be averaged to calculate the corresponding contour vertex.
        """

        #preprocessing vertex groups
        verts, faces = import_mesh_obj(template_mesh_path)

        with open(vert_grp_path, 'rb') as file:
            vert_grps = pickle.load(file=file)
            self.vert_grps = self.preprocess_some_vert_groups(vert_grps, verts)

    def preprocess_some_vert_groups(self, vert_grps, mesh_verts):
        new_vert_grps = copy.deepcopy(vert_grps)

        name = "verts_measure_front_line_string" #self.name_verts_measure_front_line_string

        shoulder_pubic_verts_idxs = new_vert_grps[name]
        new_vert_grps[name] = HumanMeasure.sort_vertical_line_string_points(shoulder_pubic_verts_idxs, mesh_verts)

        name = "verts_measure_back_line_string"
        shoulder_pubic_verts_idxs = new_vert_grps[name]
        new_vert_grps[name] = HumanMeasure.sort_vertical_line_string_points(shoulder_pubic_verts_idxs, mesh_verts)

        return new_vert_grps

    @staticmethod
    def sort_vertical_line_string_points(vert_idxs, mesh_verts):
        verts = mesh_verts[vert_idxs, 1:3]
        idxs_sorted = np.argsort(-verts[:,1])

        vert_idxs_sorted = vert_idxs[idxs_sorted]
        #verts_sorted = mesh_verts[vert_idxs_sorted, 1:3]
        #plt.axes().set_aspect(1.0)
        #plt.plot(verts_sorted[:,0], verts_sorted[:,1], '-r')
        #plt.plot(verts_sorted[0,0], verts_sorted[0,1], '+b')
        #plt.show()
        return vert_idxs_sorted

    @staticmethod
    def calc_best_bust_circumference(all_vgroups, mesh_verts, grpname_prefix):
        bust_grp_prefix = grpname_prefix
        bust_groups_pairs = list(filter(lambda item : bust_grp_prefix in item[0], all_vgroups.items()))
        bust_groups = list(map(lambda kv_pair:kv_pair[1], bust_groups_pairs))
        return HumanMeasure.find_largest_circumference(bust_groups, mesh_verts)

    @staticmethod
    def calc_best_waist_circumference(all_vgroups, mesh_verts, grpname_prefix):
        waist_grp_prefix = grpname_prefix
        waist_groups_pairs = list(filter(lambda item: waist_grp_prefix in item[0], all_vgroups.items()))
        waist_groups = list(map(lambda kv_pair:kv_pair[1], waist_groups_pairs))
        return HumanMeasure.find_smallest_circumference(waist_groups, mesh_verts)

    @staticmethod
    def calc_best_hip_circumference(all_vgroups, mesh_verts, grpname_prefix):
        hip_grp_prefix = grpname_prefix
        hip_groups_pairs = list(filter(lambda item: hip_grp_prefix in item[0], all_vgroups.items()))
        hip_groups = list(map(lambda kv_pair:kv_pair[1], hip_groups_pairs))
        return HumanMeasure.find_largest_circumference(hip_groups, mesh_verts)

    @staticmethod
    def find_best_hip_circumference(all_vgroups, mesh_verts, grpname_prefix):
        hip_grp_prefix = grpname_prefix
        hip_groups_pairs = list(filter(lambda item: hip_grp_prefix in item[0], all_vgroups.items()))
        hip_groups = list(map(lambda kv_pair:kv_pair[1], hip_groups_pairs))
        return HumanMeasure.find_largest_circumference(hip_groups, mesh_verts)

    @staticmethod
    def find_largest_circumference(vgroups, mesh_verts):
        results = list(map(lambda bust_grp : calc_point_set_circumference(mesh_verts[bust_grp,:], proj_fit_plane=False), vgroups))
        circs = [ret[0] for ret in results]
        contours = [ret[1] for ret in results]
        best_idx = np.argmax(circs)
        return circs[best_idx],  contours[best_idx]

    @staticmethod
    def find_smallest_circumference(vgroups, mesh_verts):
        results = list(map(lambda bust_grp : calc_point_set_circumference(mesh_verts[bust_grp,:], proj_fit_plane=False), vgroups))
        circs = [ret[0] for ret in results]
        contours = [ret[1] for ret in results]
        best_idx = np.argmin(circs)
        return circs[best_idx], contours[best_idx]

    @staticmethod
    def calc_front_body_length(front_line_str_vert_idxs, mesh_verts):
        """
        Measure from top of shoulder to pubic bone line
        - Top of shoulder: where neck meets shoulder, right above collar bone
        - public bone line: I am not sure. My understanding of pubic bone line is from this clip: https://www.youtube.com/watch?v=7yxZkHpAB4g
        :param front_line_str_vert_idxs:
        :param mesh_verts:
        """
        points = mesh_verts[front_line_str_vert_idxs, :]
        segs = np.diff(points)
        total_len = np.sum(np.linalg.norm(segs, 1))
        return total_len, points

    @staticmethod
    def calc_shoulder_to_bust_length(front_line_str_vert_idxs, mesh_verts, bust_z_loc):
        """
        Measure from top of shoulder to center of nipple on the same side
        - Top of shoulder: where neck meets shoulder, right above collar bone
        :param front_line_str_vert_idxs: 
        :param mesh_verts: 
        :param bust_z_loc: 
        :return: 
        """
        points = mesh_verts[front_line_str_vert_idxs, :]
        dists_to_bust_z = np.abs(points[:,2] - bust_z_loc)
        bust_point_idx =  np.argmin(dists_to_bust_z)
        shoulder_to_bust_points = points[:bust_point_idx, :]
        segs = np.diff(shoulder_to_bust_points)
        totol_len = np.sum(np.linalg.norm(segs, 1))
        return totol_len

    @staticmethod
    def calc_shoulder_to_waist_length(front_line_str_vert_idxs, mesh_verts, waist_z_loc):
        """
        Measure from top of shoulder to waist, over the bust
        - Have client put hands on waist.
        - Top of shoulder: where neck meets shoulder, right above collar bone
        :param front_line_str_vert_idxs: 
        :param mesh_verts: 
        :param waist_z_loc: 
        :return: 
        """
        points = mesh_verts[front_line_str_vert_idxs, :]
        dists_to_waist_z = np.abs(points[:,2] - waist_z_loc)
        waist_point_idx =  np.argmin(dists_to_waist_z)
        shoulder_to_waist_points = points[:waist_point_idx, :]
        segs = np.diff(shoulder_to_waist_points)
        totol_len = np.sum(np.linalg.norm(segs, 1))
        return totol_len

    @staticmethod
    def calc_waist_to_crotch_length(front_line_str_vert_idxs, mesh_verts, waist_z_loc):
        """
        Measure from waist to pubic bone line
        - Have client put hands on waist
        :param front_line_str_vert_idxs: 
        :param mesh_verts: 
        :param waist_z_loc: 
        :return: 
        """
        points = mesh_verts[front_line_str_vert_idxs, :]
        dists_to_waist_z = np.abs(points[:,2] - waist_z_loc)
        waist_point_idx =  np.argmin(dists_to_waist_z)
        waist_to_pubic_points = points[waist_point_idx:, :]
        segs = np.diff(waist_to_pubic_points)
        totol_len = np.sum(np.linalg.norm(segs, 1))
        return totol_len

    @staticmethod
    def calc_bikini_girth(front_line_str_vert_idxs, back_line_str_vert_idxs, mesh_verts, high_hip_z_loc):
        """
        At center of body, measure from front high hip line to back high hip line, through crotch
        :param front_line_str_vert_idxs: 
        :param back_line_str_vert_idxs: 
        :param mesh_verts: 
        :param high_hip_z_loc: 
        :return: 
        """
        def high_hip_to_crotch_line(line_vert_idxs):
            points = mesh_verts[line_vert_idxs, :]
            dists_to_hip_z = np.abs(points[:, 2] - high_hip_z_loc)
            hip_point_idx = np.argmin(dists_to_hip_z)
            hip_to_pubic_points = points[hip_point_idx:, :]
            segs = np.diff(hip_to_pubic_points)
            totol_len = np.sum(np.linalg.norm(segs, 1))
            return totol_len

        front_len = high_hip_to_crotch_line(front_line_str_vert_idxs)
        back_len = high_hip_to_crotch_line(back_line_str_vert_idxs)
        return front_len + back_len, np.vstack([mesh_verts[front_line_str_vert_idxs,:], mesh_verts[back_line_str_vert_idxs,:]])

    @staticmethod
    def calc_half_girth(front_line_str_vert_idxs, back_line_str_vert_idxs, mesh_verts, waist_z_loc):
        """
        At center of body, measure from front waist to back waist, through crotch.
        :param front_line_str_vert_idxs: 
        :param back_line_str_vert_idxs: 
        :param mesh_verts: 
        :param waist_z_loc: 
        :return: 
        """
        def waist_to_crotch_length(line_vert_idxs):
            points = mesh_verts[line_vert_idxs, :]
            dists_to_hip_z = np.abs(points[:, 2] - waist_z_loc)
            waist_point_idx = np.argmin(dists_to_hip_z)
            hip_to_pubic_points = points[waist_point_idx:, :]
            segs = np.diff(hip_to_pubic_points)
            totol_len = np.sum(np.linalg.norm(segs, 1))
            return totol_len

        front_len = waist_to_crotch_length(front_line_str_vert_idxs)
        back_len = waist_to_crotch_length(back_line_str_vert_idxs)
        return front_len + back_len, np.vstack([mesh_verts[front_line_str_vert_idxs,:], mesh_verts[back_line_str_vert_idxs,:]])

    @staticmethod
    def calc_full_girth(front_line_str_vert_idxs, back_line_str_vert_idxs, mesh_verts):
        """
        Measure from top of shoulder to (same) top of shoulder, measuring through crotch and over the bust
        :param front_line_str_vert_idxs: 
        :param back_line_str_vert_idxs: 
        :param mesh_verts: 
        :return: 
        """
        front_points = mesh_verts[front_line_str_vert_idxs, :]
        front_segs = np.diff(front_points)
        front_len = np.sum(np.linalg.norm(front_segs, 1))

        back_points = mesh_verts[back_line_str_vert_idxs, :]
        back_segs = np.diff(back_points)
        back_len = np.sum(np.linalg.norm(back_segs, 1))

        return front_len + back_len, np.vstack([mesh_verts[front_line_str_vert_idxs,:], mesh_verts[back_line_str_vert_idxs,:]])

    @staticmethod
    def correct_height(verts, expected_height):
        vmin = verts.min(0)
        dif = verts.max(0) - vmin
        cur_height = dif.max()
        scale = expected_height/cur_height
        verts = (verts - vmin)  * scale + vmin
        #final_height = (verts.max() - verts.min()).max()
        #print('final_height: ', final_height, " expected height: ", expected_height)
        return verts

    def measure(self, verts):
        """
        :param verts: customer vertices in form of Victoria's topology
        :return: a dict of measurement values
        - m_circ_* items refer to circumference values like bust, waist, etc.
        - m_len_* items refer to length values like upper arm, shoulder-to-wasit, etc.
        """
        assert verts.shape[0] == 49963, 'unexpected number of vertices'
        assert verts.shape[1] == 3, 'unexpected vertex format'

        landmarks = self.filter_landmarks(self.vert_grps, verts)

        #circ_contours = self._filter_circumference_contours(verts)
        #measure = self.calc_measurements(circ_contours, landmarks)
        measure, contours = self.calc_all_measurements(verts, self.vert_grps, landmarks)

        maxv = verts.max(0)
        minv = verts.min(0)
        dif = maxv - minv
        h = dif.max()
        measure['height'] = h

        return measure, contours, landmarks

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

    def vgroup_points(self, verts, vert_groups, name):
        #print(name)
        return verts[vert_groups[name], :]

    def calc_all_measurements(self, verts, vert_groups,  landmarks):
        data = {}
        data['m_breast_depth']  = calc_breast_depth(self.vgroup_points(verts, vert_groups, 'verts_measure_rbreast'))
        data['m_cross_breast']  = calc_cross_right_breast(self.vgroup_points(verts, vert_groups, 'verts_measure_rbreast'))

        bust_info  = HumanMeasure.calc_best_bust_circumference(vert_groups, verts, "verts_circ_bust_")
        #waist_info = HumanMeasure.calc_best_waist_circumference(vert_groups, verts, "verts_circ_waist_")
        waist_info = calc_point_set_circumference(self.vgroup_points(verts, vert_groups, 'verts_circ_waist'))
        #hip_info = HumanMeasure.calc_best_hip_circumference(vert_groups, verts, "verts_circ_hip_")
        hip_info = calc_point_set_circumference(self.vgroup_points(verts, vert_groups, 'verts_circ_hip'))
        data['m_circ_bust'] = bust_info
        data['m_circ_waist'] = waist_info
        data['m_circ_hip'] = hip_info

        hip_ver_z = np.mean(bust_info[1][:,2])
        waist_ver_z = np.mean(waist_info[1][:,2])

        # approximate high hip vertical location
        high_hip_ver_z = 0.7*hip_ver_z + 0.3*waist_ver_z

        data['m_len_front_body'] = HumanMeasure.calc_front_body_length(
            self.vert_grps['verts_measure_front_line_string'],
            verts)

        data['m_len_half_girth'] = HumanMeasure.calc_half_girth(
            self.vert_grps['verts_measure_front_line_string'],
            self.vert_grps['verts_measure_back_line_string'],
            verts,
            waist_ver_z)

        data['m_len_bikini_girth'] = HumanMeasure.calc_bikini_girth(
            self.vert_grps['verts_measure_front_line_string'],
            self.vert_grps['verts_measure_back_line_string'],
            verts,
            high_hip_ver_z)

        data['m_len_full_girth'] = HumanMeasure.calc_full_girth(
            self.vert_grps['verts_measure_front_line_string'],
            self.vert_grps['verts_measure_back_line_string'],
            verts)
        #
        data['m_circ_neck'] = calc_point_set_circumference(self.vgroup_points(verts, vert_groups, 'verts_circ_neck'), proj_fit_plane=True)
        data['m_circ_upperbust'] = calc_point_set_circumference(self.vgroup_points(verts, vert_groups, 'verts_circ_upperbust'))
        data['m_circ_underbust'] = calc_point_set_circumference(self.vgroup_points(verts, vert_groups, 'verts_circ_underbust'))
        data['m_circ_highhip'] = calc_point_set_circumference(self.vgroup_points(verts, vert_groups, 'verts_circ_highhip'), alpha_threshold=5)
        data['m_circ_thigh'] = calc_point_set_circumference(self.vgroup_points(verts, vert_groups, 'verts_circ_thigh'))
        data['m_circ_knee'] = calc_point_set_circumference(self.vgroup_points(verts, vert_groups, 'verts_circ_knee'))

        data['m_circ_upperarm'] = calc_point_set_circumference(self.vgroup_points(verts, vert_groups, 'verts_circ_upperarm'), proj_fit_plane=True)
        data['m_circ_elbow'] = calc_point_set_circumference(self.vgroup_points(verts, vert_groups, 'verts_circ_elbow'), proj_fit_plane=True)
        data['m_circ_wrist'] = calc_point_set_circumference(self.vgroup_points(verts, vert_groups, 'verts_circ_wrist'), proj_fit_plane=True)

        top_shoulder = np.mean(self.vgroup_points(verts, vert_groups, 'verts_measure_top_shoulder'), axis=0)
        mid_nipple = np.mean(self.vgroup_points(verts, vert_groups, 'verts_measure_mid_nipple'), axis=0)
        data['m_shoulder_to_bust'] = (np.linalg.norm(top_shoulder - mid_nipple), np.vstack([top_shoulder, mid_nipple]))

        #under_neck = landmarks['verts_ld_shoulder']
        #data['m_len_front_bodice'] = (abs(under_neck[2] - elbow[2]), np.vstack([under_neck, elbow]))

        shoulder = landmarks['verts_ld_shoulder']
        elbow = np.mean(self.vgroup_points(verts, vert_groups, 'verts_circ_elbow'), axis=0)
        data['m_len_upperarm'] = (np.linalg.norm(shoulder - elbow), np.vstack([shoulder, elbow]))

        wrist = np.mean(self.vgroup_points(verts, vert_groups, 'verts_circ_wrist'), axis=0)
        seg = np.vstack([wrist, shoulder])
        #shifting the segment outward a bit for better visualization
        dir = 0.1*(wrist-shoulder)
        shift = np.array([-dir[2], 0, dir[0]])
        seg += shift
        data['m_len_sleeve'] = (np.linalg.norm(wrist - shoulder), seg)

        knee = np.mean(self.vgroup_points(verts, vert_groups, 'verts_circ_knee'), axis=0)
        len_waist_knee = np.abs(knee[2] - waist_ver_z)
        waist_knee = knee.copy(); waist_knee[2] = waist_ver_z
        data['m_len_waist_knee'] = (len_waist_knee, np.vstack([knee, waist_knee]))


        hem = landmarks['verts_ld_hem']
        len_skirt_waist_to_hem = abs(waist_ver_z - hem[2])
        waist_hem = hem.copy(); waist_hem[2] = waist_ver_z
        segs = np.vstack([waist_hem, hem])
        #for the sake of visualization, shift the segment rightward a bit to avoid overlapping with other segmetns
        dx = 0.1*waist_ver_z
        segs[:,0] += dx
        data['m_len_skirt_waist_to_hem'] = (len_skirt_waist_to_hem, segs)

        M = dict([(name, item[0]) for name, item in data.items()])
        C = dict([(name, item[1]) for name, item in data.items()])

        return M, C

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

def precalculate_measure_info():
    dir = '/media/D1/data_1/projects/Oh/codes/human_estimation/data/meta_data_shared/'
    measure_vert_groups_path = f'{dir}/victoria_measure_vert_groups.pkl'
    mesh_path = f'{dir}/predict_sample_mesh.obj'
    calculate_contour_circ_neighbor_idxs(dir, measure_vert_groups_path, mesh_path)

import vtk
from mayavi.mlab import *
from mayavi import mlab
import tempfile
import cv2 as cv
def test_measure_viz(verts, triangles, contours, landamarks = None):
    #mlab.options.offscreen = True
    m=triangular_mesh(verts[:, 0], verts[:, 1], verts[:, 2], triangles, color=(0.2,0.4,0.8))
    mlab.view(-90, 90)
    for name, contour in contours.items():
        mlab.plot3d(contour[:,0], contour[:,1], contour[:,2], tube_radius=0.005, colormap='Spectral')
    mlab.show()

    with tempfile.TemporaryDirectory() as tmp_dir:
        #front view
        #path_f = f'{tmp_dir}/f.png'
        mlab.view(-90, 90)
        #mlab.savefig(path_f)
        mlab.show()
        #img = cv.imread(path_f)
        #img_f = img[:, 200 - 150:200 + 150, :3]
        #img_f = img_f[:,:,::-1]

        #side view
        #path_s = f'{tmp_dir}/s.png'
        #mlab.view(0, 90)
        #mlab.savefig(path_s)
        #img = cv.imread(path_s)
        #img_s = img[:, 200 - 150:200 + 150, :3]
        #img_s = img_s[:,:,::-1]

    #otherwise, it will cause leak mem
    #mlab.clf()

    #return img_f, img_s

if __name__ == '__main__':
    dir = '/media/D1/data_1/projects/Oh/codes/human_estimation/data/meta_data_shared/'
    measure_vert_groups_path = f'{dir}/victoria_measure_vert_groups.pkl'
    mesh_path = f'{dir}/predict_sample_mesh.obj'
    hmmeasure = HumanMeasure(measure_vert_groups_path, mesh_path)

    vert_dir = '/media/F/projects/Oh/data/body_test_data/body_designer_1/result_mesh/'
    name = 'IMG_20191008_152712709_front_IMG_20191008_152732578_side.obj'
    mesh = import_mesh_tex_obj(f'{vert_dir}/{name}')
    verts = mesh['v']

    measure, contours, landmarks = hmmeasure.measure(verts)
    for k, v in measure.items():
        print(k, v)

    tri_mesh_path = f'{dir}/vic_mesh_only_triangle.obj'
    _, tris = import_mesh_obj(tri_mesh_path)

    #test_measure_viz(verts, tris, contours, landmarks)
    #project_silhouette_mayavi(verts, tris, contours, ortho_proj=True, body_opacity=0.4)
    gen_measurement_color_annotation()