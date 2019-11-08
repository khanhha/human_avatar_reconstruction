import bpy
from mathutils import Vector
import math
from pathlib import Path
import pickle as pkl
import numpy as np
import math
import os
from collections import defaultdict
import shutil
import copy
import sys
import time
import pickle

scene = bpy.context.scene
g_cur_file_name = ''

def select_single_obj(obj):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select = True
    bpy.context.scene.objects.active = obj

def delete_obj(obj):
    select_single_obj(obj)
    bpy.ops.object.delete()

def transform_obj_caesar_pca(obj, s=0.01):

    select_single_obj(obj)

    bpy.ops.transform.resize(value=(s, s, s))
    bpy.ops.object.transform_apply(location=True, scale=True, rotation=True)

    mesh = obj.data
    org = Vector(mesh.vertices[7652].co[:]) #the crotch point
    org[2] = 0.0
    org[0] = 0.0
    bpy.ops.transform.translate(value=-org)
    bpy.ops.object.transform_apply(location=True, scale=False, rotation=False)

    select_single_obj(obj)
    bpy.ops.object.transform_apply(location=True, scale=True, rotation=True)


def collect_vertex_group_idxs(obj, name):
    idxs = []
    for idx, v in enumerate(obj.data.vertices):
        for vgrp in v.groups:
            grp_name = obj.vertex_groups[vgrp.group].name
            if grp_name == name:
                idxs.append(idx)
    return idxs

def find_heightest_vert_idx(obj, v_idxs):
    vert_cos = np.array([obj.data.vertices[idx].co[:] for idx in v_idxs])
    max_z_idx = int(np.argmax(vert_cos[:, 2]))
    return v_idxs[max_z_idx]

def mpii_is_correct_mesh(obj):
    n_verts = len(obj.data.vertices)
    n_faces = len(obj.data.polygons)
    if n_verts != 6449 or n_faces != 12894:
        return False
    else:
        return True

def set_realistic_img_view_mode():
    for area in bpy.context.screen.areas:  # iterate through areas in current screen
        if area.type == 'VIEW_3D':
            for space in area.spaces:  # iterate through spaces in current VIEW_3D area
                if space.type == 'VIEW_3D':  # check if space is a 3D view
                    space.viewport_shade = 'SOLID'  # set the viewport shading to rendered
                    space.use_matcap = True
                    space.show_outline_selected = False


def set_silhouette_silhouette_mode():
    for area in bpy.context.screen.areas:  # iterate through areas in current screen
        if area.type == 'VIEW_3D':
            for space in area.spaces:  # iterate through spaces in current VIEW_3D area
                if space.type == 'VIEW_3D':  # check if space is a 3D view
                    space.viewport_shade = 'TEXTURED'  # set the viewport shading to rendered
                    #space.show_outline_selected = True

def avg_co(mesh, ld_idxs):
    avg_co = Vector((0.0, 0.0, 0.0))
    for idx in ld_idxs:
        avg_co += mesh.vertices[idx].co

    avg_co /= len(ld_idxs)
    return avg_co

def collect_joint_vertex_grp_cos(obj):
    grp_names = []
    for grp in obj.vertex_groups:
        if 'joint_' in grp.name:
            grp_names.append(grp.name)

    grp_vert_idxs = {}
    for name in grp_names:
        grp_vert_idxs[name] = collect_vertex_group_idxs(obj, name)

    joints = {}
    joint_vert_idxs = {}
    for grp_name, vert_idxs in grp_vert_idxs.items():
        list_cos = []
        list_idxs = []
        for idx in vert_idxs:
            list_cos.append(obj.data.vertices[idx].co)
            list_idxs.append(idx)

        joints[grp_name] = list_cos
        joint_vert_idxs[grp_name] = list_idxs

    return joints, joint_vert_idxs

#estimate joint as the average of its vertex group
def estimate_joint_positions(obj, grp_vert_cos):
    joints = {}
    for grp_name, vert_cos in grp_vert_cos.items():
        avg = Vector((0.0, 0.0, 0.0))
        for co in vert_cos:
            avg = avg + co
        avg = avg / float(len(vert_cos))
        joints[grp_name] = avg
    return joints

#assign the estimated joint locations to the armarture object
def set_joints(obj, joints):
    #obj.select = True
    #bpy.context.scene.objects.active = obj
    select_single_obj(obj)

    bpy.ops.object.mode_set(mode='EDIT')

    t = obj.matrix_world.inverted() * joints['joint_neck']
    obj.data.edit_bones['shoulder.L'].head =  t
    obj.data.edit_bones['shoulder.R'].head = t
    obj.data.edit_bones['head'].head = t

    ####################################
    t = obj.matrix_world.inverted() * joints['joint_head']
    obj.data.edit_bones['head'].tail = t

    ####################################
    t =  obj.matrix_world.inverted() *  joints['joint_lshoulder']
    obj.data.edit_bones['upper_arm.L'].head = t
    obj.data.edit_bones["shoulder.L"].tail  = t

    t =  obj.matrix_world.inverted() *  joints['joint_rshoulder']
    obj.data.edit_bones['upper_arm.R'].head = t
    obj.data.edit_bones["shoulder.R"].tail = t

    ##################################
    t = joints['joint_lbreast']
    org_center = 0.5*(obj.data.edit_bones['breast.L'].head + obj.data.edit_bones['breast.L'].tail)
    new_center = obj.matrix_world.inverted() * t
    delta = new_center - org_center
    obj.data.edit_bones['breast.L'].head += delta
    obj.data.edit_bones['breast.L'].tail += delta

    t = joints['joint_rbreast']
    org_center = 0.5*(obj.data.edit_bones['breast.R'].head + obj.data.edit_bones['breast.R'].tail)
    new_center = obj.matrix_world.inverted() * t
    delta = new_center - org_center
    obj.data.edit_bones['breast.R'].head += delta
    obj.data.edit_bones['breast.R'].tail += delta

    ##########################
    t = obj.matrix_world.inverted() * 0.5*(joints['joint_lhip'] + joints['joint_rhip'])
    obj.data.edit_bones["spine"].head =  t
    obj.data.edit_bones["pelvis.L"].head =  t
    obj.data.edit_bones["pelvis.R"].head =  t

    #############################
    t =  obj.matrix_world.inverted() * joints['joint_upper_breast']
    obj.data.edit_bones["upper_breast"].head = t
    obj.data.edit_bones["breast"].tail = t

    #############################
    t =  obj.matrix_world.inverted() * joints['joint_under_breast']
    obj.data.edit_bones["breast"].head = t
    obj.data.edit_bones["upper_hip"].tail = t

    #############################
    t =  obj.matrix_world.inverted() * joints['joint_upper_hip']
    obj.data.edit_bones["upper_hip"].head = t
    obj.data.edit_bones["spine"].tail = t

    ##################################
    t = joints['joint_lhip']
    obj.data.edit_bones["thigh.L"].head = obj.matrix_world.inverted() * t

    t = joints['joint_rhip']
    obj.data.edit_bones["thigh.R"].head = obj.matrix_world.inverted() * t

    ############################3
    t = joints['joint_lpelvis']
    obj.data.edit_bones['pelvis.L'].tail = obj.matrix_world.inverted() * t

    t = joints['joint_rpelvis']
    obj.data.edit_bones['pelvis.R'].tail = obj.matrix_world.inverted() * t

    #############################
    t =  obj.matrix_world.inverted() * joints['joint_rknee']
    obj.data.edit_bones["thigh.R"].tail = t
    obj.data.edit_bones['shin.R'].head = t

    t = obj.matrix_world.inverted() * joints['joint_lknee']
    obj.data.edit_bones["thigh.L"].tail = t
    obj.data.edit_bones['shin.L'].head = t

    ##############################
    t = joints['joint_rankle']
    obj.data.edit_bones['shin.R'].tail = obj.matrix_world.inverted() * t

    t = joints['joint_lankle']
    obj.data.edit_bones['shin.L'].tail = obj.matrix_world.inverted() * t

    #########################
    t = obj.matrix_world.inverted() * joints['joint_relbow']
    obj.data.edit_bones["upper_arm.R"].tail = t
    obj.data.edit_bones['forearm.R'].head = t

    t = obj.matrix_world.inverted() * joints['joint_lelbow']
    obj.data.edit_bones["upper_arm.L"].tail = t
    obj.data.edit_bones['forearm.L'].head = t

    #################################
    t = joints['joint_rwrist']
    obj.data.edit_bones['forearm.R'].tail = obj.matrix_world.inverted() * t

    t = joints['joint_lwrist']
    obj.data.edit_bones['forearm.L'].tail = obj.matrix_world.inverted() * t
    ###########################

    bpy.ops.object.mode_set(mode='OBJECT')

def reset_pose(arm_obj):
    select_single_obj(arm_obj)

    bpy.ops.object.mode_set(mode='POSE')

    for pose_bone in arm_obj.pose.bones:
        pose_bone.bone.select = True
    bpy.ops.pose.rot_clear()
    for pose_bone in arm_obj.pose.bones:
        pose_bone.bone.select = False

    bpy.ops.object.mode_set(mode='OBJECT')

def rotate_front_armature_random(arm_obj):
    reset_pose(arm_obj)

    bpy.ops.object.mode_set(mode='POSE')

    arm_angle_range = [-5,20]
    leg_angle_range = [0,15]
    axis = 'Z'

    #upper arm left
    angle = np.random.rand() * (arm_angle_range[1]-arm_angle_range[0]) + arm_angle_range[0]
    pbone = arm_obj.pose.bones['upper_arm.L']
    pbone.rotation_mode='XYZ'
    pbone.rotation_euler.rotate_axis(axis, math.radians(angle))

    #upper arm right
    pbone = arm_obj.pose.bones['upper_arm.R']
    pbone.rotation_mode='XYZ'
    angle = -angle
    pbone.rotation_euler.rotate_axis(axis, math.radians(angle))

    #thigh left
    leg_delta_angle = np.random.rand() * (leg_angle_range[1]-leg_angle_range[0]) + leg_angle_range[0]
    pbone = arm_obj.pose.bones['thigh.L']
    pbone.rotation_mode='XYZ'
    leg_delta_angle = -leg_delta_angle
    pbone.rotation_euler.rotate_axis(axis, math.radians(leg_delta_angle))

    #thigh right
    pbone = arm_obj.pose.bones['thigh.R']
    pbone.rotation_mode='XYZ'
    leg_delta_angle = -leg_delta_angle
    pbone.rotation_euler.rotate_axis(axis, math.radians(leg_delta_angle))

    bpy.ops.object.mode_set(mode='OBJECT')

def rotate_side_armature_random(arm_obj):
    reset_pose(arm_obj)
    bpy.ops.object.mode_set(mode='POSE')

    axis = 'Z'

    #rotate spine
    hip_angle_range=[-10,15]
    angle = np.random.rand()*(hip_angle_range[1] - hip_angle_range[0]) + hip_angle_range[0]
    pbone = arm_obj.pose.bones['upper_hip']
    pbone.rotation_mode = 'XYZ'
    pbone.rotation_euler.rotate_axis(axis, math.radians(angle))

    #rotate neck angle
    head_angle_range = [-15,15]
    angle = np.random.rand()*(head_angle_range[1] - head_angle_range[0]) + head_angle_range[0]
    pbone = arm_obj.pose.bones['head']
    pbone.rotation_mode = 'XYZ'
    pbone.rotation_euler.rotate_axis(axis, math.radians(angle))

    bpy.ops.object.mode_set(mode='OBJECT')

def project_synthesized(verts_co_dir, out_sil_root_dir, N_pose_variants_per_mesh=20,
                        joint_path = None, n_file_to_process = -1,
                        do_front_sil = True, do_side_sil = True,
                        always_renew = False):
    """
    project front/side silhoutte for N_pose_variants_per_mesh
    :param verts_co_dir: folder containers *.npy vertex array files
    :param out_sil_root_dir: output folder contain output front/side silhouettes
    :param N_pose_variants_per_mesh: number of random pose per mesh
    :param joint_path: path to export joint vertex groups.
    :param n_file_to_process: debug. set it to -1 for prcessing all meshes
    :param do_front_sil: project front silhouette or not
    :param do_side_sil: project side silhouette or not
    :param always_renew: always renew silhouette or ignore generated ones.
    """
    #output silhouette folder
    sil_root_dir = out_sil_root_dir
    sil_f_dir = os.path.join(*[sil_root_dir , 'sil_f_raw'])
    sil_s_dir = os.path.join(*[sil_root_dir , 'sil_s_raw'])
    os.makedirs(sil_f_dir, exist_ok=True)
    os.makedirs(sil_s_dir, exist_ok=True)

    #project only front,side or both?    
    front_sil = do_front_sil
    side_sil = do_side_sil

    # the number of pose variants per mesh that we want to create
    N_pos_variants = N_pose_variants_per_mesh

    #get the template mesh 
    vic_tpl_obj = bpy.data.objects['vic_template']
    vic_tpl_mesh = vic_tpl_obj.data
    
    #get the skeleton/armature object
    armature_obj = bpy.data.objects['metarig']
    
    #get vertex indices of left and right arms
    # for side profile, we collapse left and right arm to a single point so that left/right arm
    # do not interfere with the side silhouette
    larm_v_idxs = collect_vertex_group_idxs(vic_tpl_obj, 'larm')
    rarm_v_idxs = collect_vertex_group_idxs(vic_tpl_obj, 'rarm')
    rarm_sample_v_idx = find_heightest_vert_idx(vic_tpl_obj, larm_v_idxs) #a sample vert1ex on right arm to collapse arm
    larm_sample_v_idx = find_heightest_vert_idx(vic_tpl_obj, rarm_v_idxs) #a sample vertex on left arm to collapse arm
    
    # neck landmark vertex indices for adjusing camera height.
    # for each mesh, we set camera height to the neck level.
    neck_v_idxs = collect_vertex_group_idxs(vic_tpl_obj, 'neck_landmark')
    
    # list all vertex file paths. sort it for consistent order    
    paths = sorted([path for path in Path(verts_co_dir).glob('*.npy')])
    
    # we use two cameras: one for front silhouette and one for side silhouette
    cam_front_obj = bpy.data.objects['Camera']
    cam_side_obj = bpy.data.objects['Camera_side']

    #for debugging
    #idx = np.random.randint(0, len(paths))
    #paths = [paths[idx]]
    #paths = [paths[15]]
    #paths = paths[:5]

    # export the joint vertex indices. for example. left shoulder joint vertex groups = [10, 20, 1000, 2000, ...]
    joint_grp_vert_cos, joint_vert_groups = collect_joint_vertex_grp_cos(vic_tpl_obj)
    if joint_path is not None:
        with open(joint_path, 'wb') as file:
            pickle.dump(obj=joint_vert_groups, file=file)

    #get reference to left and right arm vertex coordinates for speed optimization
    larm_cos = [vic_tpl_mesh.vertices[vi].co for vi in larm_v_idxs]
    rarm_cos = [vic_tpl_mesh.vertices[vi].co for vi in rarm_v_idxs]

    N_files = len(paths) if n_file_to_process <=0 else min(n_file_to_process, len(paths))
    for i in range(N_files):
        #time_start = time.time()
        print('prog: ', i, '/', N_files)
        path = paths[i]
        name = path.stem

        #genearte file names for all pose variants
        #object1_pose0.png, object1_pose1.png, ... object1_pose20.png
        front_sil_paths = []
        side_sil_paths = []
        processed_file = True
        for i in range(N_pos_variants):
            name_i = name + '_pose' + str(i)
            front_sil_path = os.path.join(*[sil_f_dir, name_i])
            side_sil_path = os.path.join(*[sil_s_dir, name_i])
            front_sil_paths.append(front_sil_path)
            side_sil_paths.append(side_sil_path)
            #if one of silhouette pose is missing, we considconer this obj file as non-processed  yet
            #HOWEVER: this is not 100% correct because even two front/side images exist, they could be broken file
            #it happens when we turn off blender while it is dumping an image to di sk => the image will be borken.
            #TRICK to get around: when you resume blender file, please try to delete 100 recent dumped images for safe
            #or it is recommended to run Blender until it is done.
            if not Path(front_sil_path+'.png').exists() or  not Path(side_sil_path+'.png').exists():
                processed_file = False

        #if this obj file is already processed, go on and ignore it
        #by doing it, we can turn off the blender file and resume the process at another time
        if (not always_renew) and processed_file:
           continue

        #load vertex array of the subject from disk
        verts = np.load(path)
        #print('verts.shape = ', verts.shape)

        #replace with the current vertex array
        #for vi in range(n_v):
        #    vic_tpl_mesh.vertices[vi].co[:] = verts[vi,:]
        vic_tpl_mesh.vertices.foreach_set("co", verts.flatten())

        #translate, scale the imported object
        transform_obj_caesar_pca(vic_tpl_obj, s = 10.0)

        #set camera height
        neck_ld_co = avg_co(vic_tpl_mesh, neck_v_idxs)
        cam_front_obj.location[2] = neck_ld_co[2]
        cam_side_obj.location[2] = neck_ld_co[2]

        #estimate joint location and assign joint location to the armature object
        joints = estimate_joint_positions(vic_tpl_obj, joint_grp_vert_cos)
        set_joints(armature_obj, joints)

        #select both objects and make the armature object active
        vic_tpl_obj.select = True
        armature_obj.select = True
        bpy.context.scene.objects.active = armature_obj #make the armature object active

        #rigging weight calculation: it could be disabled for fast debugging with a single object.
        bpy.ops.object.parent_set(type='ARMATURE_AUTO')

        #the file with posfix _0 is the original file
        if front_sil:
            for i in range(N_pos_variants):
                #make front camera active
                select_single_obj(cam_front_obj)
                bpy.context.scene.camera = cam_front_obj

                #hide the armature so that it won't appear in the silhouette
                armature_obj.hide = True
                #set the output front silhouette path
                bpy.data.scenes['Scene'].render.filepath = front_sil_paths[i]
                bpy.ops.render.opengl(write_still=True, view_context=True)
                armature_obj.hide = False

                #randomly rotate leg, arm
                rotate_front_armature_random(armature_obj)

            #reset pose after we're done
            reset_pose(armature_obj)

        if side_sil:
            # collapse left and right arm to avoid arm intrusion in the side profile
            larm_co = vic_tpl_mesh.vertices[larm_sample_v_idx].co[:]
            for co in larm_cos:
                co[:] = larm_co
            #for vi in larm_v_idxs:
            #    vic_tpl_mesh.vertices[vi].co[:] = larm_co

            rarm_co = vic_tpl_mesh.vertices[rarm_sample_v_idx].co[:]
            for co in rarm_cos:
                co[:] = rarm_co
            #for vi in rarm_v_idxs:
            #    vic_tpl_mesh.vertices[vi].co[:] = rarm_co

            for i in range(N_pos_variants):
                #make side camera active
                select_single_obj(cam_side_obj)
                bpy.context.scene.camera = cam_side_obj

                #hide the armature so that it won't appear in the silhouette
                armature_obj.hide = True
                #set the output side silhouette path
                bpy.data.scenes['Scene'].render.filepath = side_sil_paths[i]
                bpy.ops.render.opengl(write_still=True, view_context=True)
                armature_obj.hide = False

                rotate_side_armature_random(armature_obj)

        reset_pose(armature_obj)


#main function
#please manually adjust the folder paths to the paths on your PC. It is kind of tricky to use blender argument parsing

#set verts_co_root_dir to the correct path on our PC
#the folder contain the vertex arrays of all male/female subjects
verts_co_root_dir = '/media/D1/data_1/projects/Oh/data/3d_human/caesar_obj/vic_pca_models_syn/verts/'
obj_vert_dirs = [os.path.join(*[verts_co_root_dir, 'male']), os.path.join(*[verts_co_root_dir, 'female'])]

#set out_sil_root_dir to a folder on our PC
#the output front front/side silhouettes will be exported to this folder
out_sil_root_dir = '/home/khanhhh/Oh/blender_images/syn_test/'
os.makedirs(out_sil_root_dir, exist_ok=True)
out_sil_dirs = [os.path.join(*[out_sil_root_dir, 'male']), os.path.join(*[out_sil_root_dir, 'female'])]

# output file: exporting joint vertices for joint estimation later
# Oh said that we need the jont location so the prediction mesh should be rigged
#joint_path = '/media/D1/data_1/projects/Oh/codes/human_estimation/data/meta_data_shared/victoria_joint_vert_groups.pkl'
#it should be exported once. because we are done now, so set it to none
joint_path = None

#loop over folder for male/female
for vert_dir, out_sil_dir in zip(obj_vert_dirs, out_sil_dirs):
    print('=====================================')
    print('start processing folder: ', vert_dir, out_sil_dir)
    project_synthesized(verts_co_dir=vert_dir, out_sil_root_dir=out_sil_dir, n_file_to_process=2, always_renew=True)
    print('finish folder: ', vert_dir, out_sil_dir)
    print('=====================================')

print('finish processing for all folfders')