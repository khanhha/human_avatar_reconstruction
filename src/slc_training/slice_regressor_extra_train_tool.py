import argparse
from pathlib import Path
from slc_training.slice_def import SliceModelInputDef, SliceID
from slc_training.slice_train_util import load_slice_data_1, SlcData, load_bad_slice_names
from slc_training.slice_regressor_dtree_1 import SliceRegressor
import os
import pickle

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-slc_dir",  required=True, type=str, help="root directory contains all slice directory")
    ap.add_argument("-feature_dir", required=True, type=str, help="root directory contains all slice code directory")
    ap.add_argument("-bad_slc_dir", required=True, type=str, help="root directory contains all slice code directory")
    ap.add_argument("-model_dir", required=True, type=str, help="root directory contains all slice code directory")
    ap.add_argument("-slc_ids", required=True, type=str, help="root directory contains all slice code directory")
    ap.add_argument("-mode", required=True, type=str, help="model definition model. single or neighbor")
    args = ap.parse_args()

    ALL_SLC_DIR  = args.slc_dir
    ALL_SLC_FEATURE_DIR = args.feature_dir
    BAD_SLC_DIR = args.bad_slc_dir
    MODEL_DIR_ROOT  = os.path.join(args.model_dir, args.mode)
    train_slc_ids = args.slc_ids

    os.makedirs(MODEL_DIR_ROOT, exist_ok=True)


    all_slc_ids = [path.stem for path in Path(args.slc_dir).glob('*')]
    if train_slc_ids == 'all' or train_slc_ids == 'All':
        train_slc_ids = all_slc_ids
    else:
        train_slc_ids = train_slc_ids.split(',')
        for id in train_slc_ids:
            assert id in all_slc_ids, f'{id}: unrecognized slice id'

    #find all needed slice ids
    load_slc_ids = set()
    for slc_id in train_slc_ids:
        load_slc_ids.add(slc_id)
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

        out_slc_data = all_slc_data[train_slc_id]

        if args.mode == 'neighbor':
            print(f'\nstarting training slice {train_slc_id} in mode: {args.mode}')

            input_ids = SliceModelInputDef.get_input_def(train_slc_id)
            in_slc_data = [all_slc_data[id] for id in input_ids]

            fnames = SlcData.extract_shared_fnames(in_slc_data+ [out_slc_data])
            X, Y  = SlcData.build_training_data(in_slc_data, out_slc_data, fnames)
            #print(X.shape, Y.shape)

            net = SliceRegressor(slc_id=train_slc_id, model_input_slc_ids=input_ids)
        else:
            print(f'\nstarting training slice {train_slc_id} in mode: single')

            in_slc_data = [all_slc_data[train_slc_id]]

            fnames = SlcData.extract_shared_fnames(in_slc_data+ [out_slc_data])
            X, Y  = SlcData.build_training_data(in_slc_data, out_slc_data, fnames)

            net = SliceRegressor(slc_id=train_slc_id, model_input_slc_ids=[train_slc_id])

        train_idxs, test_idxs = net.fit(X, Y, n_jobs=10)
        model_path = os.path.join(MODEL_DIR_ROOT, f'{train_slc_id}.pkl')
        train_fnames = {fnames[idx] for idx in train_idxs}
        test_fnames = {fnames[idx] for idx in test_idxs}
        with open(model_path, 'wb') as file:
            #net.save_to_path(model_path)
            data = {'model':net, 'train_fnames': train_fnames, 'test_fnames':test_fnames}
            pickle.dump(obj=data, file=file)

