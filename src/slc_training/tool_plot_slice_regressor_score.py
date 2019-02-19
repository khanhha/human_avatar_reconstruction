import argparse
from pathlib import Path
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-model_dir",  required=True, type=str, help="root directory contains all slice directory")
    ap.add_argument("-ids",  required=True, type=str, help="root directory contains all slice directory")
    args = ap.parse_args()

    model_ids = args.ids.split(',')

    dir = os.path.join(args.model_dir, 'cluster_local_global')
    slc_ids = [path.stem for path in Path(dir).glob('*.*')]

    models_score = []
    for model_id in model_ids:

        dir = os.path.join(args.model_dir, model_id)

        model_slice_score = []
        for slc_id in slc_ids:
            model_path = os.path.join(dir, f'{slc_id}.pkl')
            with open(str(model_path),'rb') as file:
                data = pickle.load(file)
                if 'test_score' in data:
                    model_slice_score.append(data['test_score'])

        models_score.append(model_slice_score)

    fig, ax = plt.subplots()
    n_groups = len(slc_ids)
    index = np.arange(n_groups)

    bar_width = 0.15
    opacity = 0.4
    error_config = {'ecolor': '0.3'}

    colors = ['red','green','blue', 'yellow','black']
    assert len(colors) == len(model_ids)
    for idx, model_id in enumerate(model_ids):
        slc_scores = models_score[idx]
        rects1 = ax.bar(np.arange(n_groups)+idx*bar_width, slc_scores, bar_width, alpha=opacity, color=colors[idx], error_kw=error_config, label=model_id)

    ax.set_xlabel('Slices')
    ax.set_ylabel('Model R2 Scores')
    ax.set_title('Scores by slices and models')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(slc_ids)
    ax.legend()

    fig.tight_layout()
    plt.show()


