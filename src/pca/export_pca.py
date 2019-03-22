import scipy.io as io
import argparse
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-in_dir", type=str)
    args = ap.parse_args()

    in_dir = '/media/khanhhh/data_1/projects/Oh/data/3d_human/caesar/'

    mean_points = io.loadmat(f'{in_dir}/meanShape.mat')['points'] #shape=(6449,3)



