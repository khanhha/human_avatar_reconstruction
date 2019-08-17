import argparse
import logging
import os
import cv2 as cv
from pathlib import Path
import pickle

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path
from measure.pose_common import HumanPose, PosePart

logger = logging.getLogger('TfPoseEstimator')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

class IncorrectPoseFound(Exception):
    pass

class PoseExtractorTf:
    def __init__(self, resize_size = (432, 368), model_id = 'cmu', init = True):
        #if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736
        self.resize_size = resize_size
        assert model_id == 'cmu'
        self.model_id = model_id
        self.estimator = None

        if init == True:
            self.init_estimator()

    def init_estimator(self):
        if self.estimator is None:
            self.estimator = TfPoseEstimator(get_graph_path(self.model_id), target_size=self.resize_size)

    def extract_single_pose(self, img, resize_out_ratio = 4.0, debug = False):
        self.init_estimator()
        humans_ = self.estimator.inference(img, resize_to_default=(self.resize_size[0] > 0 and self.resize_size[1] > 0), upsample_size=resize_out_ratio)

        if len(humans_) < 1:
            raise IncorrectPoseFound("no human pose found")
        else:
            #TODO: in some case, one human pose are separated into two HuamNPose instances. we have to do it to merge all parts together
            pose = HumanPose(pairs = [])
            for human in humans_:
                for id, p in human.body_parts.items():
                    part = PosePart(uidx=p.uidx, part_idx=p.part_idx, x=p.x, y=p.y, score=p.score)
                    pose.body_parts[id] = part

            if debug == False:
                return pose, None
            else:
                debug_img = TfPoseEstimator.draw_humans(img, humans_, imgcopy=True)
                return pose, debug_img

import shutil
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-r", "--root_dir", required=True, help="image folder")
    ap.add_argument("-i", "--input_dir", required=True, help="image folder")
    ap.add_argument("-o", "--output_dir", required=True, help='output pose dir')
    ap.add_argument('-debug_name', required=False, type=str, default='')
    args = ap.parse_args()
    DIR_IN = os.path.join(*[args.root_dir, args.input_dir]) + '/'
    DIR_OUT = os.path.join(*[args.root_dir, args.output_dir]) + '/'

    debug_name = args.debug_name

    os.makedirs(DIR_OUT, exist_ok=True)

    extractor = PoseExtractorTf()

    error_files = []
    for img_path in Path(DIR_IN).glob('*.*'):
        if debug_name != '' and debug_name not in img_path.stem:
            continue
        print(img_path)
        img = cv.imread(str(img_path))
        try:
            pose, img_pose = extractor.extract_single_pose(img, debug=True)
            cv.imwrite(f'{DIR_OUT}/{img_path.stem}.png', img_pose)
            print(f'export pose to file {img_path}')
            with open(f'{DIR_OUT}/{img_path.stem}.pkl', 'wb') as file:
                pickle.dump(obj = pose, file=file)
        except Exception as exp:
            print('catched one exception: ', exp)
            error_files.append(img_path)
            continue

    print(f'unexpected files: ')
    for path in error_files:
        print(f'\t{path}')