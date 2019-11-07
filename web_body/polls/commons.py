import io
from PIL import Image
from torchvision import models
import torchvision.transforms as transforms
from deploy.hm_pipeline import HumanRGBModel
from deploy.hm_measurement import HumanMeasure
from deploy.data_config import config_get_data_path
import numpy as np

g_model_dir = '../cnn_run_data/models/'
g_metadata_dir = '../cnn_run_data/metadata/'

def get_body_model():
    shape_model_path = config_get_data_path(g_model_dir, 'shape_model_pytorch_joint')
    deeplab_path = config_get_data_path(g_model_dir, 'deeplab_tensorflow_model')
    vic_mesh_path  = config_get_data_path(g_model_dir, 'victoria_template_mesh')
    vic_tri_mesh_path  = config_get_data_path(g_metadata_dir, 'victoria_triangle_mesh')
    body_model = HumanRGBModel(hmshape_model_path=shape_model_path, hmsil_model_path=deeplab_path, mesh_path=vic_mesh_path, triangle_mesh_path=vic_tri_mesh_path,  use_gpu=False)
    return body_model

def get_body_measure():
    measure_vert_grps_path = config_get_data_path(g_metadata_dir, 'victoria_measure_vert_groups')
    predict_sample_mesh_path = config_get_data_path(g_metadata_dir, 'predict_sample_mesh')
    bd_measure = HumanMeasure(vert_grp_path=measure_vert_grps_path, template_mesh_path=predict_sample_mesh_path)
    return bd_measure

def get_model():
    model = models.densenet121(pretrained=True)
    model.eval()
    return model

def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


# ImageNet classes are often of the form `can_opener` or `Egyptian_cat`
# will use this method to properly format it so that we get
# `Can Opener` or `Egyptian Cat`
def format_class_name(class_name):
    class_name = class_name.replace('_', ' ')
    class_name = class_name.title()
    return class_name
