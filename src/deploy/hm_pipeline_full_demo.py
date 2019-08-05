from deploy.hm_shape_pred_pytorch_model import HmShapePredPytorchModel
from deploy.hm_shape_pred_model import HmShapePredModel
from deploy.hm_sil_pred_model import HmSilPredModel
from pca.nn_util import crop_silhouette_pair
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
from face_utils.face_extractor import FaceExtractor
from common.obj_util import import_mesh_tex_obj, export_mesh_tex_obj

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-model_dir",  required=True, type=str, help="the direction where shape_model.jlb  and deeplab model are stored")
    ap.add_argument("-meta_data_dir",  required=True, type=str, help="meta data directory")
    ap.add_argument("-in_txt_file",  required=True, type=str, help="the texture file to the input data directory. each line of the texture file have the following format:"
                                                                   "front_img  side_img  height_in_meter  gender  are_silhouette_or_not")
    ap.add_argument("-save_sil", required=False, default=False, type=bool, help="save silhouettes that are calculated from RGB input images")
    ap.add_argument("-out_dir",  required=True, type=str, help="output directory")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    #shape_model_path = os.path.join(*[args.model_dir, 'shape_model_pytorch.pt'])
    shape_model_path = os.path.join(*[args.model_dir, 'shape_model.jlb'])
    deeplab_path = os.path.join(*[args.model_dir, 'deeplabv3_xception_ade20k_train_2018_05_29.tar.gz'])
    vic_mesh_path = os.path.join(*[args.model_dir, 'vic_mesh.obj'])
    text_mesh_path = os.path.join(*[args.meta_data_dir, 'vic_mesh_textured_warped.obj'])
    tex_mesh = import_mesh_tex_obj(text_mesh_path)

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

            if len(comps) == 6:
                face_img_path = os.path.join(*[dir, comps[5]])
            else:
                face_img_path = front_img_path

            # if 'face_2.jpg' not in str(face_img_path):
            #      continue

            assert gender == 0.0 or gender == 1.0 , 'unexpected gender. just accept 1 or 0'
            assert is_sil == 0 or is_sil == 1, 'unexpected sil flag. just accept 1 or 0'
            assert height >= 1.0 and height  <= 2.5, 'unexpected height. not in range of [1.0, 2.5]'

            assert Path(front_img_path).exists() and Path(side_img_path).exists() and Path(face_img_path).exists()

            if not is_sil:
                img_f = cv.imread(front_img_path)
                img_s = cv.imread(side_img_path)
                img_face_org = cv.imread(face_img_path)

                verts, faces, sil_f, sil_s = body_model.predict(img_f, img_s, height, gender)

                #out_path = os.path.join(*[args.out_dir, f'{Path(front_img_path).stem}_org_head_{idx}.obj'])
                #export_mesh(out_path, verts=verts, faces = tex_mesh['f'])

                if args.save_sil:
                    sil_f  = (sil_f*255.0).astype(np.uint8)
                    sil_s  = (sil_s*255.0).astype(np.uint8)
                    out_sil_f_path = os.path.join(*[args.out_dir, f'{Path(front_img_path).stem}_sil.jpg'])
                    cv.imwrite(out_sil_f_path, sil_f)
                    out_sil_s_path = os.path.join(*[args.out_dir, f'{Path(side_img_path).stem}_sil.jpg'])
                    cv.imwrite(out_sil_s_path, sil_s)
                    print(f'\texported silhouette {out_sil_f_path} - {out_sil_s_path}')

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

                texture = face_texture_processor.embed(prn_remap_tex, img_face, img_face_seg, img_face_landmarks)

                out_mesh = {'v': verts, 'vt': tex_mesh['vt'], 'f': tex_mesh['f'], 'ft': tex_mesh['ft']}

                out_path = os.path.join(*[args.out_dir, f'{Path(front_img_path).stem}_{Path(face_img_path).stem}.obj'])
                export_mesh_tex_obj(out_path, out_mesh, img_tex=texture[:,:,::-1])

                # debug
                # plt.subplot(121)
                # plt.imshow(img_face)
                # plt.subplot(122)
                # plt.imshow(texture[:,:,::-1])
                # plt.savefig(os.path.join(*[args.out_dir, f'{Path(front_img_path).stem}_{Path(face_img_path).stem}_debug.jpg']))
            else:
                img_f = cv.imread(front_img_path, cv.IMREAD_GRAYSCALE)
                img_s = cv.imread(side_img_path, cv.IMREAD_GRAYSCALE)
                ret_0, img_f = cv.threshold(img_f, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
                ret_1, img_s = cv.threshold(img_s, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
                verts, faces = body_model.predict_sil(img_f, img_s, height, gender)

                out_path = os.path.join(*[args.out_dir, f'{Path(front_img_path).stem}_{idx}.obj'])
                print(f'\texported obj object to path {out_path}')
                export_mesh(out_path, verts=verts, faces=faces)