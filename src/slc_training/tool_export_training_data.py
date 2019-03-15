import argparse
from pathlib import Path
from slc_training.slice_def import SliceModelInputDef, SliceID
from slc_training.slice_train_util import load_slice_data_1, SlcData, load_bad_slice_names
import os
import pickle

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-slc_dir",  required=True, type=str, help="root directory contains all slice directory")
    ap.add_argument("-feature_dir", required=True, type=str, help="root directory contains all slice code directory")
    ap.add_argument("-bad_slc_dir", required=True, type=str, help="root directory contains all bad slice id for each type of slice")
    ap.add_argument("-out_dir", required=True, type=str, help="output directory")
    ap.add_argument("-mode", required=False, default='local_global', type=str, help="output directory")
    args = ap.parse_args()

    input_mode_def = SliceModelInputDef(args.mode)

    all_slc_ids = [path.stem for path in Path(args.slc_dir).glob('*')]
    print('\n all slice ids = ', all_slc_ids, '\n')

    #load all needed slice data
    print('start loading all slice data')
    all_slc_data = {}
    for slc_id in all_slc_ids:
        SLC_DIR = os.path.join(args.slc_dir, slc_id)
        FEATURE_DIR = os.path.join(args.feature_dir, slc_id)
        bad_fnames = load_bad_slice_names(args.bad_slc_dir, slc_id)
        data = load_slice_data_1(slc_id, SLC_DIR, FEATURE_DIR, bad_fnames)
        all_slc_data[slc_id] = data

    os.makedirs(args.out_dir, exist_ok=True)
    print(f'\nnumber of slices = {len(all_slc_ids)}\n')
    for train_slc_id in all_slc_data:
        print(f'\nstart ouput slice data of {train_slc_id}')
        out_slc_data = all_slc_data[train_slc_id]

        in_slc_ids = input_mode_def.get_input_def(train_slc_id)

        in_slc_data = [all_slc_data[id] for id in in_slc_ids]

        fnames = SlcData.extract_shared_fnames(in_slc_data + [out_slc_data])
        X,  Y  = SlcData.build_training_data(in_slc_data,  out_slc_data, fnames)
        print(f'\tinput slice ids: {in_slc_ids}')
        print(f'\tdata shape: X_shape: {X.shape}. Y_shape = {Y.shape}')

        out_path = os.path.join(*[args.out_dir, f'{train_slc_id}.pkl'])
        print(f'\toutput slice data {train_slc_id} to path {out_path}')
        with open(out_path, 'wb') as file:
            pickle.dump(obj={'X':X, 'Y':Y}, file=file)



