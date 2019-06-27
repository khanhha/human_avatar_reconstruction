import argparse
import cv2 as cv
import numpy as np

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", type=str, required=True, help='path to skin patch image')
    ap.add_argument("-o", type=str, required=True, help='path to output texture')
    args = ap.parse_args()

    tex_size = 1024

    img = cv.imread(args.i)

    G0 = img.copy()
    gpB = [G0]
    G = G0.copy()
    for i in range(10):
        G = cv.pyrDown(G)
        gpB.append(G)
        if G.shape[0] == 1:
            break
    G_highest_level = gpB[-1]

    skin_texture = np.zeros((tex_size, tex_size, 3), dtype=np.uint8)
    skin_texture[:, :, :] = G_highest_level[0, 0, :]

    cv.imwrite(args.o, skin_texture)