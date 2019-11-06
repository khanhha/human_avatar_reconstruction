import json
import numpy as np
#from .commons import get_model, transform_image, get_body_model, get_body_measure
#from common.viz_util import build_gt_predict_viz

#update
#body_model = get_body_model()
#body_measure = get_body_measure()

#model = get_model()
#imagenet_class_index = json.load(open('imagenet_class_index.json'))

def infer_mesh(img_f, img_s, height, gender):
    verts, _, sil_f, sil_s = body_model.predict(img_f, img_s, height, gender)
    verts = body_measure.correct_height(verts, height)
    return verts, body_model.tpl_triangles, sil_f, sil_s

def build_shape_visualization(verts, sil_f_rgb, sil_s_rgb, measure_contours = None, ortho_proj = None, body_opacity=1.0):
    img_body_viz = build_gt_predict_viz(verts, body_model.tpl_triangles, sil_f_rgb, sil_s_rgb, measure_contours=measure_contours, ortho_proj=ortho_proj, body_opacity=body_opacity)
    return img_body_viz

def infer_mesurement(verts, height):
    verts = body_measure.correct_height(verts, height)
    measures, contours, landmarks =  body_measure.measure(verts)
    return measures, contours

# def get_prediction(image_bytes):
#     try:
#         tensor = transform_image(image_bytes=image_bytes)
#         outputs = model.forward(tensor)
#     except Exception:
#         return 0, 'error'
#     _, y_hat = outputs.max(1)
#     predicted_idx = str(y_hat.item())
#     return imagenet_class_index[predicted_idx]
