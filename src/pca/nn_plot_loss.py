import argparse
from pathlib import Path
import torch

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-input_dir", type=str, required=True)
    args = ap.parse_args()

    for path in Path(args.input_dir).glob('*.pt'):
        if 'epoch' not in path.stem:
            continue
        #load the min loss so far
        parts = path.stem.split('_')
        epoch = int(parts[-1])
        print(epoch)
        state = torch.load(path)
        val_los = state['valid_loss']
        train_loss = state['train_loss']




