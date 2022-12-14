import argparse
from pathlib import Path
from slc_training.slice_def import SliceModelInputDef, SliceID
from slc_training.slice_train_util import load_slice_data_1, SlcData, load_bad_slice_names
from slc_training.slice_regressor_dtree_1 import SliceRegressor, SliceRegressorLocalGlobal
import os
import pickle
import numpy as np

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
    train_slc_ids = args.slc_ids

    MODEL_DIR_ROOT  = args.model_dir
    os.makedirs(MODEL_DIR_ROOT, exist_ok=True)

    global_in_slc_ids = ['Hip', 'Waist', 'Bust']

    all_slc_ids = [path.stem for path in Path(args.slc_dir).glob('*')]
    if train_slc_ids == 'all' or train_slc_ids == 'All':
        train_slc_ids = all_slc_ids
    else:
        train_slc_ids = train_slc_ids.split(',')
        for id in train_slc_ids:
            assert id in all_slc_ids, f'{id}: unrecognized slice id'

    #find all needed slice ids
    load_slc_ids = set()
    for id in global_in_slc_ids:
        load_slc_ids.add(id)
    for slc_id in train_slc_ids:
        load_slc_ids.add(slc_id)
        input_def_ids = SliceModelInputDef.get_input_def(slc_id)
        for id in input_def_ids:
            load_slc_ids.add(id)

    #load all needed slice data
    all_slc_data = {}
    for slc_id in load_slc_ids:
        SLC_DIR = os.path.join(ALL_SLC_DIR, slc_id)
        FEATURE_DIR = os.path.join(ALL_SLC_FEATURE_DIR, slc_id)
        bad_fnames = load_bad_slice_names(BAD_SLC_DIR, slc_id)
        data = load_slice_data_1(slc_id, SLC_DIR, FEATURE_DIR, bad_fnames)
        all_slc_data[slc_id] = data

    for train_slc_id in train_slc_ids:

        out_slc_data = all_slc_data[train_slc_id]

        print(f'\nstarting training slice {train_slc_id}')

        local_in_slc_ids = SliceModelInputDef.get_input_def(train_slc_id)

        in_slc_ids = local_in_slc_ids + global_in_slc_ids
        in_slc_data = [all_slc_data[id] for id in in_slc_ids]

        fnames = SlcData.extract_shared_fnames(in_slc_data + [out_slc_data])
        X,  Y  = SlcData.build_training_data(in_slc_data,  out_slc_data, fnames)
        print(f'X  shape: {X.shape}. Y_shape = {Y.shape}' )

        net = SliceRegressorLocalGlobal(slc_id=train_slc_id, model_input_slc_ids=local_in_slc_ids, model_global_input_slc_ids=global_in_slc_ids)

        train_idxs, test_idxs, train_score, test_score = net.fit(X=X, Y=Y, n_jobs=12)

        model_path = os.path.join(MODEL_DIR_ROOT, f'{train_slc_id}.pkl')
        train_fnames = {fnames[idx] for idx in train_idxs}
        test_fnames = {fnames[idx] for idx in test_idxs}
        with open(model_path, 'wb') as file:
            #net.save_to_path(model_path)
            data = {'model':net, 'train_fnames': train_fnames, 'test_fnames':test_fnames, 'train_score':train_score, 'test_score':test_score}
            pickle.dump(obj=data, file=file)

