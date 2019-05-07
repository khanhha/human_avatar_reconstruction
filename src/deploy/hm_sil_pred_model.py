import tensorflow as tf
import cv2 as cv
import urllib
import os
import sys
import tarfile
from six.moves import urllib
import numpy as np
import time
from matplotlib import pyplot as plt

class NoSilhouetteFound(Exception):
    pass

class HmSilPredModel():

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    def __init__(self, use_mobile_model = False, use_gpu = False, save_model_path = '../data/deeplab_model/'):

        # @param ['mobilenetv2_coco_voctrainaug', 'mobilenetv2_coco_voctrainval', 'xception_coco_voctrainaug', 'xception_coco_voctrainval']
        self.use_mobile_model = use_mobile_model
        self.use_gpu = use_gpu
        self.save_model_path = save_model_path

        graph_path = self.load_model()

        self.construct_tf_session(graph_path)

    def load_model(self):
        if self.use_mobile_model == True:
            self.MODEL_NAME = 'mobilenetv2_coco_voctrainval'
        else:
            #self.MODEL_NAME = 'xception_coco_voctrainaug'
            self.MODEL_NAME = 'deeplabv3_xception_ade20k_train_2018_05_29'


        _DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'
        _MODEL_URLS = {
            'mobilenetv2_coco_voctrainaug':
                'deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz',
            'mobilenetv2_coco_voctrainval':
                'deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz',
            'xception_coco_voctrainaug':
                'deeplabv3_pascal_train_aug_2018_01_04.tar.gz',
            'xception_coco_voctrainval':
                'deeplabv3_pascal_trainval_2018_01_04.tar.gz',
            'resnet':
                'resnet_v1_101_2018_05_04.tar.gz',
            'deeplabv3_xception_ade20k_train_2018_05_29':
                'deeplabv3_xception_ade20k_train_2018_05_29.tar.gz'
        }

        _TARBALL_NAME = f'{self.MODEL_NAME}.tar.gz'
        if not os.path.exists(self.save_model_path):
            os.makedirs(self.save_model_path)

        download_path = os.path.join(self.save_model_path, _TARBALL_NAME)
        if not os.path.isfile(download_path):
            print(f'downloading deeplab model {_TARBALL_NAME} , this might take a while...')
            urllib.request.urlretrieve(_DOWNLOAD_URL_PREFIX + _MODEL_URLS[self.MODEL_NAME], download_path)
            print('download completed! loading DeepLab model...')

        return download_path

    def construct_tf_session(self, tarball_path):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()

        graph_def = None
        # Extract frozen graph from tar archive.
        tar_file = tarfile.open(tarball_path)
        for tar_info in tar_file.getmembers():
          if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
            file_handle = tar_file.extractfile(tar_info)
            graph_def = tf.GraphDef.FromString(file_handle.read())
            break

        tar_file.close()

        if graph_def is None:
          raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
          tf.import_graph_def(graph_def, name='')

        self.use_gpu = self.use_gpu
        if self.use_gpu:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            # for a gpu of 11 GB. choose 0.08 for PC model, 0.05 for a mobile model
            # TODO: initialize this value for each type of GPU
            config.gpu_options.per_process_gpu_memory_fraction = 0.5
            self.sess = tf.Session(graph=self.graph, config=config)
        else:
            config = tf.ConfigProto(device_count={'GPU': 0})
            self.sess = tf.Session(graph=self.graph, config=config)

    def is_precise_model(self):
        return not self.use_mobile_model

    def run(self, image):
        """Runs inference on a single image.

        Args:
          image: A PIL.Image object, raw input image.

        Returns:
          resized_image: RGB image resized from original input image.
          seg_map: Segmentation map of `resized_image`.
        """
        height, width = image.shape[:2]
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = cv.resize(image, target_size, interpolation=cv.INTER_AREA)
        #with tf.Session(graph=self.graph) as sess:
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]
        return resized_image, seg_map

    def extract_silhouette_aug(self, img):
        sil_0 = self.extract_silhouette(img)
        img_1 = cv.flip(img, flipCode=1)
        sil_1 = self.extract_silhouette(img_1)
        sil_1 = cv.flip(sil_1, flipCode=1)
        sil = cv.bitwise_or(sil_0, sil_1)
        return sil

    def extract_silhouette(self, img):
        _, seg_map = self.run(img)
        #test = (seg_map == 13) | (seg_map == 133)
        # values = np.unique(seg_map[:])
        # for v in values:
        #     m = seg_map==v
        #     print(v)
        #     plt.imshow(resized_im)
        #     plt.imshow(m, alpha=0.8)
        #     plt.title(f'id = {v}')
        #     plt.show()
        # plt.subplot(121)
        # plt.imshow(resized_im)
        # plt.imshow(seg_map, alpha=0.9)
        # plt.subplot(122)
        # plt.imshow(resized_im)
        # plt.imshow(test, alpha=0.9)
        # plt.show()

        #deeplab coco
        #silhouette_mask = (seg_map == 15)

        #deeplab ade20k
        #TODO: why 13 and 133? we need to investigate the id hierachy of ADA20K
        silhouette_mask = (seg_map == 13) | (seg_map == 133)

        if np.sum(silhouette_mask[:]) <= 0:
            raise NoSilhouetteFound('segmentation map: no object of type 13 and 133 are found')

        silhouette = silhouette_mask.astype(np.uint8) * 255
        silhouette = cv.morphologyEx(silhouette, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (7, 7)))

        nb_components, output, stats, centroids = cv.connectedComponentsWithStats(silhouette, connectivity=4)
        sizes = stats[:, -1]
        max_label = 1
        max_size = sizes[0]
        for i in range(nb_components):
            if sizes[i] > max_size:
                max_label = i
                max_size = sizes[i]

        silhouette_1 = np.zeros(output.shape, np.uint8)
        silhouette_1[output == max_label] = 255
        #silhouette_1 = cv.morphologyEx(silhouette_1, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT, (5, 5)))
        silhouette_1 = cv.resize(silhouette_1, img.shape[:2][::-1], cv.INTER_NEAREST)

        return silhouette_1
