import numpy as np
import cv2 as cv
import sys
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import os

g_bad_slc_names = set()
g_cur_idx = 0
g_all_paths = []

g_fig, g_ax = plt.subplots()
img_ax = None

def update_draw():
    global  img_ax
    path = g_all_paths[g_cur_idx]
    img = cv.imread(str(path))
    if img_ax is None:
        img_ax = g_ax.imshow(img)
    else:
        img_ax.set_data(img)
    bad = False
    if path.stem in g_bad_slc_names:
        bad = True
    g_ax.set_title(f'file_name = {path.stem}, idx = {g_cur_idx}, is_bad = {bad}')
    g_fig.canvas.draw()

def press(event):
    global  g_cur_idx
    if event.key == 'c':
        g_bad_slc_names.add(g_all_paths[g_cur_idx].stem)
        update_draw()
    elif event.key == 'z':
        g_bad_slc_names.remove(g_all_paths[g_cur_idx].stem)
        update_draw()
    elif event.key == 'right':
        g_cur_idx += 1
        g_cur_idx = g_cur_idx % len(g_all_paths)
        update_draw()
    elif event.key == 'left':
        g_cur_idx -= 1
        g_cur_idx = max(0, g_cur_idx)
        update_draw()
    else:
        print('unrecognized key', file=sys.stderr)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--in_dir", required=True, help="slice obj directory")
    ap.add_argument("-f", "--file", required=True, help="directory for expxorting control mesh slices")
    args = vars(ap.parse_args())

    IN_IMG_DIR  = args['in_dir']
    OUT_TXT_FILE = args['file']

    if os.path.exists(OUT_TXT_FILE):
        with open(OUT_TXT_FILE, 'r') as file:
            for name in file.readlines():
                g_bad_slc_names.add(name)

    g_all_paths = [path for path in Path(IN_IMG_DIR).glob('*.*')]
    g_cur_idx = 0
    if len(g_all_paths) > 0:
        g_fig.canvas.mpl_connect('key_press_event', press)
        g_ax.set_aspect(1.0)
        g_ax.axis('off')
        update_draw()
        plt.show()

        with open(OUT_TXT_FILE, 'w') as file:
            for name in g_bad_slc_names:
                file.write(f'{name}\n')

        print(f'wrote to file {OUT_TXT_FILE}')
    else:
        print('no file found', file=sys.stderr)
