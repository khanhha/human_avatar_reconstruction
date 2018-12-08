import sklearn
import argparse
import pickle
from src.caesar_rbf_net import RBFNet
from sklearn.tree import export_graphviz
import shutil
import os
from pathlib import Path
from dtreeviz.trees import *
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--in_path", required=True, help="slice obj directory")
    ap.add_argument("-d", "--debug_dir", required=True, help="slice obj directory")
    args = vars(ap.parse_args())

    MODEL_PATH = Path(args['in_path'])

    DEBUG_DIR_PARENT = args['debug_dir']
    DEBUG_DIR = f'{DEBUG_DIR_PARENT}/tree_viz/{MODEL_PATH.stem}'
    shutil.rmtree(DEBUG_DIR, ignore_errors=True)
    os.makedirs(DEBUG_DIR)

    model = RBFNet.load_from_path(MODEL_PATH)
    use_graphviz = False
    if use_graphviz:
        for i, tree in enumerate(model.regressor.estimators_):
            export_graphviz(tree, out_file=f'{DEBUG_DIR}/estimator_{i}.dot')
        for path in Path(DEBUG_DIR).glob('*.dot'):
            print(path)
            os.system(f'dot -Tpng {str(path)} -o {DEBUG_DIR}/{path.stem}.png')
    else:
        for i, forest in enumerate(model.regressor.estimators_):
            for j, tree in enumerate(forest.estimators_):
                viz = dtreeviz(tree,
                               X_train,
                               Y_train,
                               feature_names=[str(i) for i in range(self.n_cluster)],
                               target_name='hello')
                viz.view()
