import numpy as np
import argparse
import sys
from pathlib import Path
from src.caesar_rbf_net import RBFNet

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--in_dir", required=True, help="input meta data file")
    ap.add_argument("-m", "--model", required=True, help="input meta data file")
    ap.add_argument("-ids", "--model_ids", required=True, help="input meta data file")

    args = vars(ap.parse_args())

    SLC_DIR  = args['in_dir']
    MODEL_DIR  = args['model']
    ids  = args['model_ids'].split(',')

    for path in Path(MODEL_DIR).glob('*.pkl'):
        if len(ids) > 0 and path.stem not in ids:
            continue

        net = RBFNet.load_from_path(path)

