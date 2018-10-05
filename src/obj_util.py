import numpy as np

def export_vertices(fpath, obj):
    with open(fpath, 'w') as f:
        for i in range(obj.shape[0]):
            co = tuple(obj[i, :])
            f.write("v %.4f %.4f %.4f \n" % co)

def load_vertices(fpath):
    coords = []
    with open(fpath, 'r') as obj:
        file = obj.read()
        lines = file.splitlines()
        for line in lines:
            elem = line.split()
            if elem:
                if elem[0] == 'v':
                    coords.append((float(elem[1]), float(elem[2]), float(elem[3])))
    return np.array(coords)


