import numpy as np
from pathlib import Path
from numpy.linalg import norm
import common.util as util
import shapely.geometry as geo
from shapely.ops import nearest_points
import pickle
from scipy.spatial import KDTree
from copy import copy
from copy import deepcopy
import sys
from slc_training.slice_def import SliceID

def slice_id_3d_2d_mappings():
    mappings = {}
    mappings['LAnkle'] = 'Ankle'
    mappings['RAnkle'] = 'Ankle'

    mappings['RCalf'] = 'Calf'
    mappings['LCalf'] = 'Calf'

    mappings['RKnee'] = 'Knee'
    mappings['LKnee'] = 'Knee'

    mappings['LAux_Knee_UnderCrotch_0'] = 'Aux_Knee_UnderCrotch_0'
    mappings['LAux_Knee_UnderCrotch_1'] = 'Aux_Knee_UnderCrotch_1'
    mappings['LAux_Knee_UnderCrotch_2'] = 'Aux_Knee_UnderCrotch_2'
    mappings['LAux_Knee_UnderCrotch_3'] = 'Aux_Knee_UnderCrotch_3'
    mappings['RAux_Knee_UnderCrotch_0'] = 'Aux_Knee_UnderCrotch_0'
    mappings['RAux_Knee_UnderCrotch_1'] = 'Aux_Knee_UnderCrotch_1'
    mappings['RAux_Knee_UnderCrotch_2'] = 'Aux_Knee_UnderCrotch_2'
    mappings['RAux_Knee_UnderCrotch_3'] = 'Aux_Knee_UnderCrotch_3'

    mappings['RUnderCrotch'] = 'UnderCrotch'
    mappings['LUnderCrotch'] = 'UnderCrotch'

    mappings['Crotch'] = 'Crotch'
    mappings['Aux_Crotch_Hip_0'] = 'Aux_Crotch_Hip_0'
    mappings['Aux_Crotch_Hip_1'] = 'Aux_Crotch_Hip_1'
    mappings['Aux_Crotch_Hip_2'] = 'Aux_Crotch_Hip_2'

    mappings['Hip'] = 'Hip'
    mappings['Aux_Hip_Waist_0'] = 'Aux_Hip_Waist_0'
    mappings['Aux_Hip_Waist_1'] = 'Aux_Hip_Waist_1'
    mappings['Waist'] = 'Waist'
    mappings['Aux_Waist_UnderBust_0'] = 'Aux_Waist_UnderBust_0'
    mappings['Aux_Waist_UnderBust_1'] = 'Aux_Waist_UnderBust_1'
    mappings['Aux_Waist_UnderBust_2'] = 'Aux_Waist_UnderBust_2'
    mappings['UnderBust'] = 'UnderBust'
    mappings['Aux_UnderBust_Bust_0'] = 'Aux_UnderBust_Bust_0'
    mappings['Bust'] = 'Bust'
    mappings['Armscye'] = 'Armscye'
    mappings['Aux_Armscye_Shoulder_0'] = 'Aux_Armscye_Shoulder_0'
    mappings['Shoulder'] = 'Shoulder'
    mappings['Collar'] = 'Collar'
    mappings['Neck'] = 'Neck'

    return mappings

def breast_part_slice_id_3d_2d_mappings():
    id_2ds = ['Breast_Depth_Aux_UnderBust_Bust_0', 'Breast_Depth_Bust', 'Breast_Depth_Aux_Bust_Armscye_0']
    mappings = {}
    for id_2d in id_2ds:
        mappings['R' + id_2d] = id_2d

    for id_2d in id_2ds:
        mappings['L' + id_2d] = id_2d
    return mappings

def bone_lengths(arm_2d, ratio):
    lengths = {}
    lengths['Shin'] = ratio*norm(arm_2d['LAnkle'] - arm_2d['LKnee'])
    lengths['Thigh'] = ratio*norm(arm_2d['LHip'] - arm_2d['LKnee'])
    lengths['Torso'] = ratio*norm(arm_2d['MidHip'] - arm_2d['Neck'])
    lengths['Shoulder'] = ratio*norm(arm_2d['LShoulder'] - arm_2d['Neck'])
    lengths['UpperArm'] = ratio*norm(arm_2d['LShoulder'] - arm_2d['LElbow'])
    lengths['ForeArm'] = ratio*norm(arm_2d['LElbow'] - arm_2d['LWrist'])
    return lengths

def scale_tpl_armature(arm_3d, arm_2d, ratio):
    #leg
    blengths = bone_lengths(arm_2d, ratio)
    arm_3d['LKnee'] = arm_3d['LAnkle'] + blengths['Shin'] * util.normalize(arm_3d['LKnee'] - arm_3d['LAnkle'])
    arm_3d['LHip']  = arm_3d['LKnee'] + blengths['Thigh'] * util.normalize(arm_3d['LHip'] - arm_3d['LKnee'])

    midhip = 0.5*(arm_3d['LHip'] + arm_3d['RHip'])
    arm_3d['Neck'] = midhip + blengths['Torso'] * util.normalize(arm_3d['Neck'] - midhip)
    #TODO: Crotch, Spine, Chest

    arm_3d['LShoulder'] = arm_3d['Neck'] + blengths['Shoulder'] * util.normalize(arm_3d['LShoulder'] - arm_3d['Neck'])

    arm_3d['LElbow'] = arm_3d['LShoulder'] + blengths['UpperArm'] * util.normalize(arm_3d['LElbow'] - arm_3d['LShoulder'])

    arm_3d['LWrist'] = arm_3d['LElbow'] + blengths['ForeArm'] * util.normalize(arm_3d['LWrist'] - arm_3d['LElbow'])

    return arm_3d

def infer_arm_slice_locations(shoulder_slc_loc, neck_shoulder_len, armature, upper_arm_len, under_arm_len):
    neck_shoulder_dir = util.normalize(armature['LShoulder'] - armature['LNeck'])
    shoulder = shoulder_slc_loc + neck_shoulder_len * neck_shoulder_dir

    upper_arm_dir = util.normalize(armature['LElbow'] - armature['LShoulder'])
    under_arm_dir = util.normalize(armature['LWrist'] - armature['LElbow'])

    elbow = shoulder + upper_arm_dir * upper_arm_len
    wrist = elbow + under_arm_dir * under_arm_len
    locs = {}
    locs['LAux_Shoulder_Elbow_0'] = shoulder + 0.5 * (elbow-shoulder)
    locs['LElbow'] = elbow
    locs['LAux_Elbow_Wrist_0'] = elbow + 0.5 * (wrist - elbow)
    locs['LWrist'] = wrist
    return locs

def calc_arm_slice_locations(tpl_3d_armature, target_2d_armature, ratio):
    shoulder_len = norm(target_2d_armature['Neck'] - target_2d_armature['LShoulder'])
    shoulder_len = ratio * shoulder_len

def subdivide_catmull_temp(mesh):
    verts = mesh['verts']
    faces = mesh['faces']
    edges = mesh['edges']
    loops = mesh['loops'] #l[0]: vertex, l[1]: edge

    nv = len(verts)
    nf = len(faces)
    ne = len(edges)

    e_f = mesh['e_f']
    v_f = mesh['v_f']
    v_e = mesh['v_e']
    f_l = mesh['f_l']

    new_vert_pnts = np.zeros(shape=(nv,3), dtype=np.float32)
    new_edge_pnts = np.zeros(shape=(ne,3), dtype=np.float32)
    new_face_pnts = np.zeros(shape=(nf,3), dtype=np.float32)

    for f_idx in range(nf):
        co = np.zeros(shape=(3,), dtype=np.float32)
        f = faces[f_idx]
        for vf_idx in f:
            co += verts[vf_idx, :]
        co /= len(f)
        new_face_pnts[f_idx, :] = co

    for e_idx in range(ne):
        co = np.zeros(shape=(3,))
        for ef_idx in e_f[e_idx]:
            co += new_face_pnts[ef_idx,:]

        ev0 = edges[e_idx][0]
        ev1 = edges[e_idx][1]
        co += verts[ev0,:]
        co += verts[ev1,:]
        new_edge_pnts[e_idx, :] = co/(2.0 + len(e_f[e_idx]))

    for v_idx in range(nv):
        co_avg_f = np.zeros(shape=(3,))
        co_avg_e = np.zeros(shape=(3,))

        nvf = len(v_f[v_idx])
        for vf_idx in v_f[v_idx]:
            co_avg_f += new_face_pnts[vf_idx, :]
        co_avg_f /= nvf

        nve = len(v_e[v_idx])
        for ve_idx in v_e[v_idx]:
            co_avg_e += new_edge_pnts[ve_idx, :]
        co_avg_e /= nve

        #not a boundary vertex
        if nve == nvf:
            new_vert_pnts[v_idx, :] = (co_avg_f + 2.0*co_avg_e + (nve - 3.0) * verts[v_idx, :]) / nve
        else:
            new_vert_pnts[v_idx, :] = verts[v_idx,:]

    #create new topology
    new_vert_pnts = np.concatenate([new_vert_pnts, new_face_pnts, new_edge_pnts], axis=0)
    new_faces = []
    for f_idx in range(nf):
        nl = len(f_l[f_idx])
        for idx in range(nl):
            l_0 = loops[f_l[f_idx][idx]]
            l_1 = loops[f_l[f_idx][(idx + 1) % nl]]
            v0 = nv + nf + l_0[1]
            v1 = l_1[0]
            v2 = nv + nf + l_1[1]
            v3 = nv + f_idx
            new_faces.append((v0, v1, v2, v3))

    new_mesh = {'verts':new_vert_pnts, 'faces': new_faces}

    return new_mesh

def projec_mesh_onto_mesh(mesh_0, mesh_0_vert_idxs, mesh_1):
    verts_0 = mesh_0['verts']
    nv_0 = len(verts_0)

    faces_1 = mesh_1['faces']
    verts_1 = mesh_1['verts']
    nf_1 = len(faces_1)

    centroid_faces_1 = np.zeros(shape=(nf_1, 3), dtype=np.float)
    for f_idx in range(nf_1):

        co = np.zeros(shape=(3,), dtype=np.float32)
        for v_idx in faces_1[f_idx]:
            co += verts_1[v_idx,:]
        co /= len(faces_1[f_idx])

        centroid_faces_1[f_idx, :] = co

    # out_path = f'{OUT_DIR}{mdata_path.stem}_debug_centroid_faces.obj'
    # print(f'\toutput deformed mesh to {out_path}')
    # export_mesh(out_path, centroid_faces_1, [])

    # out_path = f'{OUT_DIR}{mdata_path.stem}_debug_mesh_0.obj'
    # print(f'\toutput deformed mesh to {out_path}')
    # export_mesh(out_path, verts_0, [])

#    tree = cKDTree(centroid_faces_1, leafsize=5)
    tree = KDTree(centroid_faces_1, leafsize=5)

    if mesh_0_vert_idxs is None:
        mesh_0_vert_idxs = range(nv_0)

    for v_idx in mesh_0_vert_idxs:
        v_co = verts_0[v_idx,:]
        #query closest faces
        _, idxs = tree.query(v_co, 5)
        cls_dst = np.inf
        cls_pnt = v_co
        for f_idx in idxs:
            f = faces_1[f_idx]
            assert len(f) == 4
            on_quad_p = util.closest_on_quad_to_point_v3(v_co, verts_1[f[0], :], verts_1[f[1], :], verts_1[f[2], :], verts_1[f[3], :])
            dst = norm(on_quad_p - v_co)
            if dst < cls_dst:
                cls_dst = dst
                #cls_pnt = 0.25*(verts_1[f[0],:] + verts_1[f[1],:] + verts_1[f[2],:] + verts_1[f[3],:])
                cls_pnt = on_quad_p
        verts_0[v_idx, :] = cls_pnt

    #out_path = f'{OUT_DIR}{mdata_path.stem}_debug_projected.obj'
    #print(f'\toutput deformed mesh to {out_path}')
    #export_mesh(out_path, np.array(test_points), [])

#the basis origin of armature is the midhip position
def transform_non_vertical_slice(slice, loc, loc_after, radius):
    slice_n = slice - loc
    rads = norm(slice_n, axis=1)
    max_rad = np.max(rads)
    scale = radius / max_rad
    slice_n *= scale
    return loc_after + slice_n

def scale_vertical_slice(slice, w_ratio, d_ratio, scale_center = None):
    if scale_center is None:
        scale_center = np.mean(slice, axis=0)
    nslice = slice - scale_center
    nslice[:,0] *= w_ratio
    nslice[:,1] *= d_ratio
    nslice = nslice + scale_center
    return nslice

def transform_arm_slices(mesh, slc_id_locs, slc_id_vert_idxs, arm_3d):
    slc_org = slc_id_locs['Slice_LElbow']
    slc_idxs = slc_id_vert_idxs['Slice_LElbow']
    slice = ctl_mesh['verts'][slc_idxs]
    radius = h_ratio * 0.5 * seg_dst_f['Elbow']
    mesh['verts'][slc_idxs, :] = transform_non_vertical_slice(slice, slc_org, arm_3d['LElbow'], radius)

    slc_org = slc_id_locs['Slice_LWrist']
    slc_idxs = slc_id_vert_idxs['Slice_LWrist']
    slice = ctl_mesh['verts'][slc_idxs]
    wrist_radius = h_ratio * 0.5 * 0.8 * seg_dst_f['Elbow']
    mesh['verts'][slc_idxs, :] = transform_non_vertical_slice(slice, slc_org, arm_3d['LWrist'], wrist_radius)

    lhand_idxs = []
    for id, idxs in slc_id_vert_idxs.items():
        if 'LHand' in id:
            lhand_idxs.append(idxs[:])
    displacement = arm_3d['LWrist'] - slc_id_locs['Slice_LWrist']
    mesh['verts'][lhand_idxs, :] += displacement

#note: this function just works for breast slice
def scale_breast_height(brst_slc, slc_height):
    #hack
    #find two ending points of the breast slice
    #1. the point with smallest x
    end_0_idx = np.argmin(brst_slc[:,0])
    #2. the point with lagrest y
    end_1_idx = np.argmax(brst_slc[:,1])
    brst_hor_seg = geo.LineString([brst_slc[end_0_idx,:2], brst_slc[end_1_idx,:2]])

    #3: breast highest height
    cur_brst_height = 0.0
    for i in range(brst_slc.shape[0]):
        if i != end_0_idx and i != end_1_idx:
            p = brst_slc[i,:2]
            dst = geo.Point(p).distance(brst_hor_seg)
            cur_brst_height = max(dst, cur_brst_height)

    ratio =  slc_height/cur_brst_height

    for i in range(brst_slc.shape[0]):
        if i != end_0_idx and i != end_1_idx:
            p = brst_slc[i,:2]
            proj = nearest_points(brst_hor_seg, geo.Point(p))[0]
            proj = np.array(proj).flatten()
            p_scaled = proj + ratio*(p-proj)
            brst_slc[i,:2] = p_scaled

    brst_slc[end_0_idx, 2] = brst_slc[end_0_idx, 2] + 0.01
    brst_slc[end_1_idx, 2] = brst_slc[end_1_idx, 2] - 0.01

    return brst_slc, end_0_idx, end_1_idx

def fix_crotch_hip_cleavage(slice, depth):
    delta = 0.05*depth
    clv_idx = 0
    slice[clv_idx, 1] = slice[clv_idx, 1] - delta
    return slice

class ControlMeshPredictor():

    slc_ids = [
            SliceID.Neck,
            SliceID.Collar,

            SliceID.Shoulder,
            SliceID.Aux_Armscye_Shoulder_0,
            SliceID.Armscye,
            SliceID.Aux_Bust_Armscye_0,

            SliceID.Bust,
            SliceID.Aux_UnderBust_Bust_0,
            SliceID.UnderBust,

            SliceID.Aux_Waist_UnderBust_2,
            SliceID.Aux_Waist_UnderBust_1,
            SliceID.Aux_Waist_UnderBust_0,

            SliceID.Waist,
            SliceID.Aux_Hip_Waist_0,
            SliceID.Aux_Hip_Waist_1,

            SliceID.Hip,
            SliceID.Aux_Crotch_Hip_2,
            SliceID.Aux_Crotch_Hip_1,
            SliceID.Aux_Crotch_Hip_0,
            SliceID.Crotch,

            SliceID.UnderCrotch,
            SliceID.Aux_Knee_UnderCrotch_3,
            SliceID.Aux_Knee_UnderCrotch_2,
            SliceID.Aux_Knee_UnderCrotch_1,
            SliceID.Aux_Knee_UnderCrotch_0,
            SliceID.Knee,

            SliceID.Calf,
            SliceID.Ankle
    ]

    def __init__(self, MODEL_DIR):
        self.models = {}
        for path in Path(MODEL_DIR).glob('*.pkl'):
            with open(str(path),'rb') as file:
                data = pickle.load(file=file)
                model = data['model']
                assert self.is_valid_model(model), f'model {model} does not have expected interface'
                id = SliceID.find_enum(model.slc_id)
                assert id is not None, f'invalid model name {model.slc_id}'
                self.models[id] = model

    def is_valid_model(self, model):
        a = hasattr(model, 'slc_id')
        b = hasattr(model, 'slc_model_input_ids')
        c = hasattr(model, 'predict')

        found = False
        for slc_id in self.slc_ids:
            if model.slc_id == slc_id.name:
                found = True

        assert found == True, 'invalid model slice id'
        d = model.slc_id == id

        return a and b and c and d

    def set_control_mesh(self, ctl_mesh, slc_id_vert_idxs, slc_id_locs, ctl_sym_vert_pairs, arm_3d_tpl):
        self.ctl_mesh = ctl_mesh
        self.slc_id_vert_idxs = slc_id_vert_idxs
        self.slc_id_locs = slc_id_locs
        self.arm_3d_tpl = arm_3d_tpl

        #make leg slices zero-centered
        for id, slc_idxs in self.slc_id_vert_idxs.items():
            is_leg = util.is_leg_contour(id)
            if is_leg:
                slice = self.ctl_mesh['verts'][slc_idxs]
                #loc = self.slc_id_locs[id]
                x_mid = 0.5*(np.min(slice[:,0]) + np.max(slice[:,0]))
                slice[:,0] = slice[:,0] - x_mid
                self.ctl_mesh['verts'][slc_idxs] = slice

        self.ctl_sym_vert_pairs = ctl_sym_vert_pairs #symmetry information. for leg and arm slices, we just predict the left side

        print('control  mesh: nverts = {0}, ntris = {1}'.format(self.ctl_mesh['verts'].shape[0],
                                                                len(self.ctl_mesh['faces'])))

    def set_template_mesh(self, tpl_mesh, tpl_height):
        self.tpl_mesh = tpl_mesh
        self.tpl_height = tpl_height
        print('victoria mesh: nverts = {0}, ntris = {1}'.format(self.tpl_mesh['verts'].shape[0],
                                                                len(self.tpl_mesh['faces'])))

    def collect_slice_data(self, seg_dst_f, seg_dst_s, seg_locs_f, seg_locs_s):
        slc_w_d= {}
        slc_locs_s = {}
        slc_locs_f = {}
        for slc_id in self.slc_ids:
            width = seg_dst_f[slc_id.name]
            depth = seg_dst_s[slc_id.name]
            slc_w_d[slc_id] = (width, depth)

            slc_locs_f[slc_id] = seg_locs_f[slc_id.name]
            slc_locs_s[slc_id] = seg_locs_s[slc_id.name]

        return slc_w_d, slc_locs_f, slc_locs_s

    def predict_1(self, seg_dst_f, seg_dst_s, seg_locs_s, seg_locs_f, height):

        slc_w_d, slc_locs_f, slc_locs_s = self.collect_slice_data(seg_dst_f, seg_dst_s, seg_locs_f, seg_locs_s)

        return self._predict(slc_w_d, slc_locs_f, slc_locs_s)

    def _predict(self, slc_w_d, slc_locs_f, slc_locs_s):
        ctl_new_mesh = deepcopy(self.ctl_mesh)
        ctl_ankle_loc = self.slc_id_locs[SliceID.Ankle.name]

        h_ratio = 1.0
        for slc_id in self.slc_ids:
            slc_idxs = self.slc_id_vert_idxs[slc_id.name]
            slice = self.ctl_mesh['verts'][slc_idxs]

            w = slc_w_d[slc_id][0] * h_ratio
            d = slc_w_d[slc_id][1] * h_ratio

            #slice location, which is the back point of the slice (point on the back slice), relative to ankle in the side image.
            slc_loc_side_img = slc_locs_s[slc_id]
            slc_loc_y = slc_loc_side_img[0]
            slc_loc_z = np.abs(slc_loc_side_img[1])

            #transform to victoria's scale. why?
            slc_loc_z = slc_loc_z * h_ratio
            slc_loc_y = slc_loc_y * h_ratio

            slc_loc_front_img = slc_locs_f[slc_id]
            slc_loc_x = slc_loc_front_img[0]
            slc_loc_x = slc_loc_x * h_ratio

            slice_out = copy(slice)

            if slc_id in self.models:
                #print('\t applied ', id_2d)
                model = self.models[slc_id]
                ratio = w/d
                pred = model.predict(np.reshape(ratio, (1,1)))[0, :]
                res_contour = util.reconstruct_contour_fourier(pred)

                slice_out[:,0] =  res_contour[1,:]
                slice_out[:,1] =  res_contour[0,:]

                # we apply x,y scaling to make sure that our the final slice match width/height measurement
                dim_range = np.max(slice_out, axis=0) - np.min(slice_out, axis=0)
                w_ratio = w / dim_range[0]
                d_ratio = d / dim_range[1]
                slice_out = scale_vertical_slice(slice_out, w_ratio, d_ratio)

                #debug
                import matplotlib.pyplot as plt
                plt.axes().set_aspect(1.0)
                plt.plot(slice_out[:, 0], slice_out[:, 1], '-r')
                plt.plot(slice_out[:, 0], slice_out[:, 1], '+r')
                plt.show()

                # hack. the slices from hip to crothc is a bit flat along cleavage. we push the cleavage vertices a bit inside
                if util.is_torso_contour(slc_id.name):
                    if slc_id.name == 'Hip' or 'Crotch' in slc_id.name:
                        slice_out = fix_crotch_hip_cleavage(slice_out, d)

                # align slice in vertical direction
                slice_out[:, 2] = slc_loc_z + ctl_ankle_loc[2]

                # align slice in horizontal direction
                # Warning: need to be careful here. we assume that the maximum point on hor dir is on the back side of Victoria's mesh
                # TODO => how to make horizontal direction consistent, when the back-to-front direction in image is opposite to back-to-front of Victoria?
                slice_y_anchor = np.max(slice_out[:, 1])
                slice_out[:, 1] += (slc_loc_y - slice_y_anchor + ctl_ankle_loc[1])

                # arrange leg slice
                if util.is_leg_contour(slc_id.name):
                    slice_out[:, 0] += slc_loc_x

                ctl_new_mesh['verts'][slc_idxs, :] = slice_out

            # for the right vertices (right leg, right arm), mirror the left vertices
            verts = ctl_new_mesh['verts']
            for pair in self.ctl_sym_vert_pairs:
                mirror_co = deepcopy(verts[pair[0]])
                mirror_co[0] = -mirror_co[0]
                verts[pair[1]] = mirror_co

            # we create two versions of the control mesh
            # the triangle version is used for deformation algorithm
            ctl_mesh_tri_new = deepcopy(self.ctl_mesh)
            ctl_mesh_tri_new['verts'] = deepcopy(ctl_new_mesh['verts'])

            return ctl_mesh_tri_new

    def predict(self, seg_dst_f, seg_dst_s, seg_locs_s, seg_locs_f, height):
        id_mappings = slice_id_3d_2d_mappings()

        ctl_new_mesh = deepcopy(self.ctl_mesh)

        #hack: the background z value extracted from image is not exact. therefore, we consider ankle z as the z starting point
        ctl_ankle_loc = self.slc_id_locs['LAnkle']

        # h_ratio = self.tpl_height / height
        h_ratio = 1.0

        #slice location in relative to ankle location in side image
        for id_3d, id_2d in id_mappings.items():
            #ignore the right body part
            if id_3d[0] == 'R':
                continue

            if id_3d not in self.slc_id_vert_idxs:
                print(f'indices of {id_3d} are not available', file=sys.stderr)
                continue

            slc_idxs = self.slc_id_vert_idxs[id_3d]
            slice = self.ctl_mesh['verts'][slc_idxs]

            if id_2d not in seg_dst_f:
                print(f'measurement of {id_2d} is not available', file=sys.stderr)
                continue

            if id_2d not in seg_locs_s:
                print(f'side location of {id_2d} is not available. ignore this slice', file=sys.stderr)
                continue

            if id_2d not in seg_locs_f:
                print(f'front location of {id_2d} is not available. ignore this slice', file=sys.stderr)
                continue

            #slice width, height
            w = seg_dst_f[id_2d]
            d = seg_dst_s[id_2d]
            w = w * h_ratio
            d = d * h_ratio

            #slice location, which is the back point of the slice (point on the back slice), relative to ankle in the side image.
            slc_loc_side_img = seg_locs_s[id_2d]
            slc_loc_y = slc_loc_side_img[0]
            slc_loc_z = np.abs(slc_loc_side_img[1])
            #transform to victoria's scale. why?
            slc_loc_z = slc_loc_z * h_ratio
            slc_loc_y = slc_loc_y * h_ratio

            slc_loc_front_img = seg_locs_f[id_2d]
            slc_loc_x = slc_loc_front_img[0]
            slc_loc_x = slc_loc_x * h_ratio

            slice_out = copy(slice)

            if id_2d in self.models:
                #print('\t applied ', id_2d)
                model = self.models[id_2d]
                ratio = w/d
                pred = model.predict(np.reshape(ratio, (1,1)))[0, :]
                res_contour = util.reconstruct_contour_fourier(pred)

                slice_out[:,0] =  res_contour[1,:]
                slice_out[:,1] =  res_contour[0,:]

                # if id_2d == 'Aux_Crotch_Hip_0' or id_2d == 'Crotch':
                #     import matplotlib.pyplot as plt
                #     plt.axes().set_aspect(1.0)
                #     plt.plot(slice_out[:,0], slice_out[:,1], '-b')
                #     plt.plot(slice_out[:,0], slice_out[:,1], '+r')
                #     plt.title(id_2d)


            #we apply x,y scaling to make sure that our the final slice match width/height measurement
            dim_range = np.max(slice_out, axis=0) - np.min(slice_out, axis=0)
            w_ratio = w / dim_range[0]
            d_ratio = d / dim_range[1]
            slice_out = scale_vertical_slice(slice_out, w_ratio, d_ratio)
            # if id_2d == 'Aux_Crotch_Hip_0' or id_2d == 'Crotch':
            #     import matplotlib.pyplot as plt
            #     plt.axes().set_aspect(1.0)
            #     plt.plot(slice_out[:, 0], slice_out[:, 1], '-r')
            #     plt.plot(slice_out[:, 0], slice_out[:, 1], '+r')
            #     plt.show()

            #hack. the slices from hip to crothc is a bit flat along cleavage. we push the cleavage vertices a bit inside
            if util.is_torso_contour(id_2d):
                if id_2d == 'Hip' or 'Crotch' in id_2d:
                    slice_out = fix_crotch_hip_cleavage(slice_out, d)

            #align slice in vertical direction
            slice_out[:, 2] = slc_loc_z + ctl_ankle_loc[2]

            #align slice in horizontal direction
            #Warning: need to be careful here. we assume that the maximum point on hor dir is on the back side of Victoria's mesh
            #TODO => how to make horizontal direction consistent, when the back-to-front direction in image is opposite to back-to-front of Victoria?
            slice_y_anchor = np.max(slice_out[:,1])
            slice_out[:, 1] += (slc_loc_y - slice_y_anchor + ctl_ankle_loc[1])

            #arrange leg slice
            if util.is_leg_contour(id_2d):
                slice_out[:, 0] += slc_loc_x

            ctl_new_mesh['verts'][slc_idxs, :] = slice_out

        #for the right vertices (right leg, right arm), mirror the left vertices
        verts = ctl_new_mesh['verts']
        for pair in self.ctl_sym_vert_pairs:
            mirror_co = deepcopy(verts[pair[0]])
            mirror_co[0] = -mirror_co[0]
            verts[pair[1]] = mirror_co

        #we create two versions of the control mesh
        #the triangle version is used for deformation algorithm
        ctl_mesh_tri_new = deepcopy(self.ctl_mesh)
        ctl_mesh_tri_new['verts'] = deepcopy(ctl_new_mesh['verts'])

        return ctl_mesh_tri_new

