import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from pathlib import Path
from common.obj_util import import_mesh_obj, export_mesh
from deploy.hm_pipeline import HumanRGBModel
from deploy.hm_head_model import HmHeadModel
from deploy.hm_face_warp import HmFPrnNetFaceTextureEmbedder
from deploy.hm_measurement import HumanMeasure, HmJointEstimator
from face_utils.face_extractor import FaceExtractor
from common.obj_util import import_mesh_tex_obj, export_mesh_tex_obj
from deploy.data_config import config_get_data_path
import pickle
from common.viz_util import build_gt_predict_viz

def export_measurement_file(file_path, measurements):
    with open(file_path, 'wt') as file:
        for name, value in measurements.items():
            file.writelines([f'{name} : {value}\n'])

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-model_dir",  required=True, type=str, help="the direction where shape_model.jlb  and deeplab model are stored")
    ap.add_argument("-meta_data_dir",  required=True, type=str, help="meta data directory")
    ap.add_argument("-in_txt_file",  required=True, type=str, help="the texture file to the input data directory. each line of the texture file have the following format:"
                                                                   "front_img  side_img  height_in_meter  gender  are_silhouette_or_not")
    ap.add_argument("-save_sil", required=False, default=False, type=bool, help="save silhouettes that are calculated from RGB input images")
    ap.add_argument("-out_dir",  required=True, type=str, help="output directory")
    ap.add_argument("-out_measure_dir", required=False, type=str, help="output directory to contain measurements", default='')
    ap.add_argument("-out_joint_dir", required=False, type=str, help="output directory to contain joints", default='')
    ap.add_argument("-out_viz_dir", required=False, type=str, help="output directory to contain joints", default='')
    ap.add_argument("-color", required=False, action="store_true", help="use body part color segmentation or not")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    #shape_model_path = os.path.join(*[args.model_dir, 'shape_model.jlb'])
    #shape_model_path = config_get_data_path(args.model_dir, 'shape_model_s')
    shape_model_path = config_get_data_path(args.model_dir, 'shape_model_pytorch_s')
    #deeplab_path = os.path.join(*[args.model_dir, 'deeplabv3_xception_ade20k_train_2018_05_29.tar.gz'])
    deeplab_path = config_get_data_path(args.model_dir, 'deeplab_tensorflow_model')
    #vic_mesh_path = os.path.join(*[args.model_dir, 'vic_mesh.obj'])
    vic_mesh_path  = config_get_data_path(args.model_dir, 'victoria_template_mesh')
    vic_tri_mesh_path  = config_get_data_path(args.meta_data_dir, 'victoria_triangle_mesh')
    _, vic_tris = import_mesh_obj(vic_tri_mesh_path)

    #text_mesh_path = os.path.join(*[args.meta_data_dir, 'vic_mesh_textured_warped.obj'])
    text_mesh_path = config_get_data_path(args.meta_data_dir, 'victoria_template_textured_mesh')
    tex_mesh = import_mesh_tex_obj(text_mesh_path)

    predict_sample_mesh_path = config_get_data_path(args.meta_data_dir, 'predict_sample_mesh')

    measure_vert_grps_path = config_get_data_path(args.meta_data_dir, 'victoria_measure_vert_groups')
    measure_circ_neighbor_idxs_path = config_get_data_path(args.meta_data_dir, 'victoria_measure_contour_circ_neighbor_idxs')
    bd_measure = HumanMeasure(vert_grp_path=measure_vert_grps_path, template_mesh_path=predict_sample_mesh_path)

    joint_vert_groups_path = config_get_data_path(args.meta_data_dir, 'victoria_joint_vert_groups')
    joint_estimator = HmJointEstimator(joint_vert_groups_path)

    args = ap.parse_args()
    assert Path(shape_model_path).exists() and Path(deeplab_path).exists()

    body_model = HumanRGBModel(hmshape_model_path=shape_model_path, hmsil_model_path=deeplab_path, mesh_path=vic_mesh_path)

    head_model = HmHeadModel(meta_dir=args.meta_data_dir, model_dir=args.model_dir)

    face_extractor = FaceExtractor(model_dir=args.model_dir)

    face_texture_processor = HmFPrnNetFaceTextureEmbedder(meta_dir=args.meta_data_dir)

    with open(args.in_txt_file, 'rt') as file:
        dir = Path(args.in_txt_file).parent

        for idx, l in enumerate(file.readlines()):
            if l[0] == '#':
                continue
            l = l.replace('\n','')
            comps = l.split(' ')
            if len(comps) != 5 and len(comps) != 6:
                print("ignore line: ", comps, f'len(comps) = {len(comps)}')
                continue

            print(comps)

            front_img_path = os.path.join(*[dir, comps[0]])
            side_img_path  = os.path.join(*[dir, comps[1]])
            height = float(comps[2])
            gender = float(comps[3])
            is_sil = int(comps[4])
            f_name = Path(front_img_path).stem

            if len(comps) == 6:
                face_img_path = os.path.join(*[dir, comps[5]])
            else:
                face_img_path = front_img_path

            assert gender == 0.0 or gender == 1.0 , 'unexpected gender. just accept 1 or 0'
            assert is_sil == 0 or is_sil == 1, 'unexpected sil flag. just accept 1 or 0'
            assert height >= 1.0 and height  <= 2.5, 'unexpected height. not in range of [1.0, 2.5]'

            assert Path(front_img_path).exists() and Path(side_img_path).exists() and Path(face_img_path).exists()

            if not is_sil:
                img_f = cv.imread(front_img_path)
                img_s = cv.imread(side_img_path)
                img_face_org = cv.imread(face_img_path)

                body_part_seg_input = args.color
                if not body_part_seg_input:
                    verts, faces, sil_f, sil_s = body_model.predict(img_f, img_s, height, gender, correct_sil_f=False, correct_sil_s=False)

                    if args.save_sil:
                        #sil_f  = (sil_f*255.0).astype(np.uint8)
                        #sil_s  = (sil_s*255.0).astype(np.uint8)
                        out_sil_f_path = os.path.join(*[args.out_dir, f'{Path(front_img_path).stem}_sil.jpg'])
                        cv.imwrite(out_sil_f_path, sil_f)
                        out_sil_s_path = os.path.join(*[args.out_dir, f'{Path(side_img_path).stem}_sil.jpg'])
                        cv.imwrite(out_sil_s_path, sil_s)
                        print(f'\texported silhouette {out_sil_f_path} - {out_sil_s_path}')
                else:
                    print('use body part segmentation input')
                    #predict 3d mesh from body part color segmentation
                    plt.subplot(121)
                    plt.imshow(img_f[:,:,::-1])
                    plt.subplot(122)
                    plt.imshow(img_s[:,:,::-1])
                    plt.show()
                    verts, faces = body_model.predict_rgb(img_f, img_s, height, gender)

                img_face, img_face_landmarks, img_face_seg = face_extractor.extract(img_face_org)
                #from BGR to RGB
                img_face = img_face[:,:,::-1]

                verts, prn_remap_tex  = head_model.predict(customer_df_verts=verts, image_rgb_front=img_face, face_landmarks=img_face_landmarks)

                # debug code
                # from deploy import hm_face_warp
                # hm_face_warp.G_debug_id = Path(face_img_path).stem
                # tmp_dir ='/home/khanhhh/data_1/projects/Oh/data/face/google_front_faces/tmp_data/'
                # data={'prn_remap_tex':prn_remap_tex, 'img_face':img_face, 'img_face_landmarks':img_face_landmarks}
                # import pickle
                # with open(f'{tmp_dir}/face_tmp_dat.pkl', 'wb') as file:
                #     pickle.dump(obj=data, file=file)
                # exit()

                skin_colors = HmFPrnNetFaceTextureEmbedder.estimate_skin_color(img_face, img_face_seg)
                # we just take the best one
                assert len(skin_colors) == 1
                best_color = skin_colors[0]
                texture = face_texture_processor.embed(prn_remap_tex, img_face, img_face_seg, img_face_landmarks, best_color, fix_nostril=True)

                out_mesh = {'v': verts, 'vt': tex_mesh['vt'], 'f': tex_mesh['f'], 'ft': tex_mesh['ft']}

                out_path = os.path.join(*[args.out_dir, f'{Path(front_img_path).stem}_{Path(side_img_path).stem}.obj'])
                export_mesh_tex_obj(out_path, out_mesh, img_tex=texture[:,:,::-1])

                if args.out_viz_dir != '':
                    os.makedirs(args.out_viz_dir, exist_ok=True)
                    sil_f_rgb = np.zeros((sil_f.shape[0], sil_f.shape[1], 3), dtype=np.uint8)
                    for i in range(3):
                        sil_f_rgb[:,:,0] = sil_f
                    sil_s_rgb = np.zeros((sil_s.shape[0], sil_s.shape[1], 3), dtype=np.uint8)
                    for i in range(3):
                        sil_s_rgb[:,:,0] = sil_s

                    viz_img = build_gt_predict_viz(verts, vic_tris, sil_f_rgb, sil_s_rgb)
                    cv.imwrite(f'{args.out_viz_dir}/{f_name}.png', viz_img[:,:,::-1])

                if args.out_measure_dir != '':
                    os.makedirs(args.out_measure_dir, exist_ok=True)
                    measure_path = os.path.join(*[args.out_measure_dir, f'{f_name}.txt'])
                    measurements = bd_measure.measure(verts, height, correct_height=True)
                    export_measurement_file(measure_path, measurements)

                if args.out_joint_dir != '':
                    os.makedirs(args.out_joint_dir, exist_ok=True)
                    joint_path = os.path.join(*[args.out_joint_dir, f'{f_name}.pkl'])
                    joints = joint_estimator.estimate_joints(verts)
                    with open(joint_path, 'wb') as file:
                        pickle.dump(obj=joints, file=file)
            else:
                img_f = cv.imread(front_img_path, cv.IMREAD_GRAYSCALE)
                img_s = cv.imread(side_img_path, cv.IMREAD_GRAYSCALE)
                ret_0, img_f = cv.threshold(img_f, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
                ret_1, img_s = cv.threshold(img_s, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
                verts, faces = body_model.predict_sil(img_f, img_s, height, gender)

                out_path = os.path.join(*[args.out_dir, f'{Path(front_img_path).stem}.obj'])
                print(f'\texported obj object to path {out_path}')
                export_mesh(out_path, verts=verts, faces=faces)