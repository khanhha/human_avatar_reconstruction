import argparse
from pathlib import Path
from slc_training.slice_def import SliceModelInputDef, SliceID
from slc_training.slice_train_util import load_slice_data_1, SlcData, load_bad_slice_names
from slc_training.slice_regressor_dtree import RBFNet
import os

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-slc_dir",  required=True, type=str, help="root directory contains all slice directory")
    ap.add_argument("-feature_dir", required=True, type=str, help="root directory contains all slice code directory")
    ap.add_argument("-bad_slc_dir", required=True, type=str, help="root directory contains all slice code directory")
    ap.add_argument("-model_dir", required=True, type=str, help="root directory contains all slice code directory")
    ap.add_argument("-slc_ids", required=True, type=str, help="root directory contains all slice code directory")
    args = ap.parse_args()

    ALL_SLC_DIR  = args.slc_dir
    ALL_SLC_FEATURE_DIR = args.feature_dir
    BAD_SLC_DIR = args.bad_slc_dir
    MODEL_DIR_ROOT  = args.model_dir
    train_slc_ids = args.slc_ids

    all_slc_ids = [slc_enum.name for slc_enum in SliceID]
    if train_slc_ids == 'all' or train_slc_ids == 'All':
        train_slc_ids = all_slc_ids
    else:
        train_slc_ids = train_slc_ids.split(',')
        for id in train_slc_ids:
            assert id in all_slc_ids, f'{id}: unrecognized slice id'

    #find all needed slice ids
    load_slc_ids = set()
    for slc_id in train_slc_ids:
        input_def_ids = SliceModelInputDef.get_input_def(slc_id)
        for id in input_def_ids:
            load_slc_ids.add(id)

    #load slice data
    all_slc_data = {}
    for slc_id in load_slc_ids:
        SLC_DIR = os.path.join(ALL_SLC_DIR, slc_id)
        FEATURE_DIR = os.path.join(ALL_SLC_FEATURE_DIR, slc_id)
        bad_fnames = load_bad_slice_names(BAD_SLC_DIR, slc_id)
        data = load_slice_data_1(slc_id, SLC_DIR, FEATURE_DIR, bad_fnames)
        all_slc_data[slc_id] = data

    for train_slc_id in train_slc_ids:

        input_ids = SliceModelInputDef.get_input_def(train_slc_id)
        in_slc_data = [all_slc_data[id] for id in input_ids]

        out_slc_data = all_slc_data[train_slc_id]

        X, Y = SlcData.build_training_data(in_slc_data, out_slc_data)
        print(X.shape, Y.shape)

        net = RBFNet(slc_id=train_slc_id, n_cluster=12, n_output=38, no_regress_at_outputs=None, debug_mode=None)
        net.fit(X, Y)

