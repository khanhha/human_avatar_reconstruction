import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from pathlib import Path
import numpy as np

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
import pandas as pd

def calc_measurement_comparsion(gt_mdata, predict_person_mdata):
    def get_gt_measure(person_id, measure_id):
        if person_id in gt_mdata.columns:
            measure_col = gt_mdata[person_id]
            measure_idx = gt_mdata['measure_id'] == measure_id
            measure = measure_col[measure_idx]
            if len(measure) == 1:
                return float(measure)
            else:
                return np.nan
        else:
            return np.nan

    def map_result(pred_row):
        person_id = pred_row['person_id']
        out_row = copy.deepcopy(pred_row)
        for k, v in pred_row.items():
            if 'm_' in k:
                measure_id = k
                pred_m = v
                pred_m *= 100 #cm => m
                gt_m = get_gt_measure(person_id, measure_id)
                if not np.isnan(gt_m):
                    error = 100*abs(pred_m-gt_m)/gt_m
                    out_row[k] = f'{gt_m:7.1f}|{pred_m:7.1f}|{error:7.1f}%'
                else:
                    out_row[k] = f'unknown|{pred_m:7.1f}|unknown'

        return out_row

    result_compare = predict_person_mdata.apply(map_result, axis=1)
    person_ids = result_compare['person_id']
    front_images = result_compare['front_image']
    result_compare = result_compare.drop('person_id', axis='columns')
    result_compare = result_compare.drop('front_image', axis='columns')
    result_compare.insert(0, "front_image", front_images)
    result_compare.insert(0, "person_id", person_ids)
    return result_compare

def load_data_txt_file(path):
    datalist = []
    with open(path, 'rt') as file:
        dir = path.parent
        dir_img = os.path.join(*[dir, 'images'])
        for idx, l in enumerate(file.readlines()):
            if l[0] == '#':
                continue
            l = l.replace('\n','')
            comps = l.split(' ')
            if len(comps) != 5 and len(comps) != 6:
                print("ignore line: ", comps, f'len(comps) = {len(comps)}')
                continue

            front_img_path = os.path.join(*[dir_img, comps[0]])
            side_img_path  = os.path.join(*[dir_img, comps[1]])
            height = float(comps[2])
            gender = float(comps[3])
            is_sil = int(comps[4])

            if len(comps) == 6:
                face_img_path = os.path.join(*[dir_img, comps[5]])
            else:
                face_img_path = front_img_path

            assert gender == 0.0 or gender == 1.0 , 'unexpected gender. just accept 1 or 0'
            assert is_sil == 0 or is_sil == 1, 'unexpected sil flag. just accept 1 or 0'
            assert height >= 1.0 and height  <= 2.5, 'unexpected height. not in range of [1.0, 2.5]'

            assert Path(front_img_path).exists() and Path(side_img_path).exists() and Path(face_img_path).exists()

            datalist.append([front_img_path, side_img_path, face_img_path, height, gender, is_sil])

    data = pd.DataFrame(datalist, columns=["front_image", "side_image", "face_image", "height", "gender", "is_silhouette"])
    return data

def load_data_xlsx_file(path):
    data = pd.read_excel(path, sheet_name='images')
    dir_img = os.path.join(*[path.parent, 'images'])

    def fix_missing_face_img(x):
        if pd.isna(x['face_image']):
            x['face_image'] = x['front_image']
        return x
    data  = data.apply(fix_missing_face_img, axis=1)

    data['face_image'] = [os.path.join(*[dir_img, name]) for name in data['face_image']]
    data['front_image'] = [os.path.join(*[dir_img, name]) for name in data['front_image']]
    data['side_image'] = [os.path.join(*[dir_img, name]) for name in data['side_image']]

    data['is_silhouette'] = data['is_silhouette'].apply(lambda x :  0 if pd.isna(x) else 1)
    data['gender'] = data['gender'].apply(lambda x : 0 if x == 'women' else 1)

    for idx, row in data.iterrows():
        assert Path(row['front_image']).exists()
        assert Path(row['side_image']).exists()
        assert Path(row['face_image']).exists()

    return data

import warnings
def load_measurement_data(path, person_ids):
    mdata = pd.read_excel(path, sheet_name='measurement')
    for person_id in person_ids:
        if person_id not in mdata:
            warnings.warn(f'measurement data for person_id "{person_id}" is no available')

    return mdata

def load_data_file(fpath):
    path = Path(fpath)
    if ".txt" in path.suffix:
        data=load_data_txt_file(path)
        return data
    elif ".xlsx" in path.suffix:
        data = load_data_xlsx_file(path)
        return data
    else:
        print('unsupport formmat')
        assert False

import copy
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

    test = pd.DataFrame()
    measures = None
    for i in range(10):
        r = {'m_circ_neck':np.random.rand(),
             'm_circ_bust':np.random.rand(),
             'm_circ_underbust':np.random.rand(),
             'm_half_girth' : np.random.rand(),
             'person_id': np.random.choice(["oh", 'kori', 'aysa']),
             'front_image': "fdfdf"}
        if measures is None:
            measures = dict(map(lambda it : (it[0], [it[1]]), r.items()))
        else:
            measures = dict(map(lambda it : (it[0], measures[it[0]]+[it[1]]), r.items()))
    measures = pd.DataFrame.from_dict(measures)

    data = load_data_file(args.in_txt_file)
    if 'person_id' in data:
        gt_mdata = load_measurement_data(args.in_txt_file, data['person_id'])
    else:
        gt_mdata = None

    pred_measurements = None
    pred_measurements_df = pd.DataFrame()

    #shape_model_path = os.path.join(*[args.model_dir, 'shape_model.jlb'])
    #shape_model_path = config_get_data_path(args.model_dir, 'shape_model_s')
    shape_model_path = config_get_data_path(args.model_dir, 'shape_model_pytorch_joint')
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

    for idx, row in data.iterrows():
        front_img_path = row['front_image']
        side_img_path  = row['side_image']
        face_img_path  = row['face_image']
        height = row['height']
        gender = row['gender']
        is_sil = row['is_silhouette']
        f_name = Path(front_img_path).stem
        if 'person_id' in row:
            person_id = row['person_id']
        else:
            person_id = Path(front_img_path).stem

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

            skin_colors = HmFPrnNetFaceTextureEmbedder.estimate_skin_color(img_face, img_face_seg)
            # we just take the best one
            assert len(skin_colors) == 1
            best_color = skin_colors[0]
            texture = face_texture_processor.embed(prn_remap_tex, img_face, img_face_seg, img_face_landmarks, best_color, fix_nostril=True)

            out_mesh = {'v': verts, 'vt': tex_mesh['vt'], 'f': tex_mesh['f'], 'ft': tex_mesh['ft']}

            out_path = os.path.join(*[args.out_dir, f'{Path(front_img_path).stem}_{Path(side_img_path).stem}.obj'])
            export_mesh_tex_obj(out_path, out_mesh, img_tex=texture[:,:,::-1])
        else:
            img_f = cv.imread(front_img_path)
            img_s = cv.imread(side_img_path)
            #ret_0, img_f = cv.threshold(img_f, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
            #ret_1, img_s = cv.threshold(img_s, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
            sil_f = cv.cvtColor(img_f, cv.COLOR_BGR2GRAY)
            sil_s = cv.cvtColor(img_s, cv.COLOR_BGR2GRAY)
            verts, faces = body_model.predict_sil(sil_f, sil_s, height, gender)

            out_path = os.path.join(*[args.out_dir, f'{Path(front_img_path).stem}.obj'])
            print(f'\texported obj object to path {out_path}')
            export_mesh(out_path, verts=verts, faces=faces)

        if args.out_viz_dir != '':
            os.makedirs(args.out_viz_dir, exist_ok=True)
            sil_f_rgb = np.zeros((sil_f.shape[0], sil_f.shape[1], 3), dtype=np.uint8)
            for i in range(3):
                sil_f_rgb[:, :, 0] = sil_f
            sil_s_rgb = np.zeros((sil_s.shape[0], sil_s.shape[1], 3), dtype=np.uint8)
            for i in range(3):
                sil_s_rgb[:, :, 0] = sil_s

            viz_img = build_gt_predict_viz(verts, vic_tris, sil_f_rgb, sil_s_rgb)
            cv.imwrite(f'{args.out_viz_dir}/{f_name}.png', viz_img[:, :, ::-1])

        if args.out_measure_dir != '':
            os.makedirs(args.out_measure_dir, exist_ok=True)
            measure_path = os.path.join(*[args.out_measure_dir, f'{f_name}.txt'])
            measurements = bd_measure.measure(verts, height, correct_height=True)
            measurements['person_id'] = person_id
            measurements['front_image'] = Path(front_img_path).stem

            if pred_measurements is None:
                # (key, value) => (key, [value])
                pred_measurements = dict(map(lambda it : (it[0], [it[1]]), measurements.items()))
            else:
                #concatenate new data for each key
                # (key, [value0, value1]) => (key, [value0, value1, value2])
                pred_measurements = dict(map(lambda it : (it[0], pred_measurements[it[0]]+[it[1]]), measurements.items()))

        if args.out_joint_dir != '':
            os.makedirs(args.out_joint_dir, exist_ok=True)
            joint_path = os.path.join(*[args.out_joint_dir, f'{f_name}.pkl'])
            joints = joint_estimator.estimate_joints(verts)
            with open(joint_path, 'wb') as file:
                pickle.dump(obj=joints, file=file)

    if args.out_measure_dir != '' and gt_mdata is not None:
        pred_measurements_df = pd.DataFrame.from_dict(pred_measurements)
        out_mpath = os.path.join(*[args.out_measure_dir, 'measurement_true_vs_predict.xlsx'])
        output = calc_measurement_comparsion(gt_mdata, pred_measurements_df)

        sheet_name = 'true_vs_predict'
        writer = pd.ExcelWriter(out_mpath, engine='xlsxwriter')
        output.to_excel(writer, sheet_name=sheet_name)
        worksheet = writer.sheets[sheet_name]
        n_cols = len(output.columns)
        worksheet.set_column(0, n_cols, 20)
        writer.save                                                                                                                                                                                                                                                                                                                                                                                                                                     ()

