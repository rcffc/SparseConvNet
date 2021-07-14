import plyfile
import numpy as np
from scipy.ndimage.measurements import label

# Map relevant classes to {0,1,...,19}, and ignored classes to -100
remapper = np.ones(150)*(-100)
for i, x in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]):
    remapper[x] = i


def print_class_distribution(labels):
    for l in np.unique(labels):
        print(str(l) + ': ' + str(len(labels[labels==l])))

path = '/igd/a4/homestud/pejiang/ScanNet/scans/scene0444_00/scene0444_00_vh_clean_2.labels.ply'
a = plyfile.PlyData().read(path)
w = np.array(a.elements[0]['label'])
w_remapped = remapper[np.array(a.elements[0]['label'])]
print_class_distribution(w_remapped)

path = '/igd/a4/homestud/pejiang/scenes/multi/0444_00/0444_00_400_scaled_normalized_predicted.ply'
a = plyfile.PlyData().read(path)
w = np.array(a.elements[0]['label'])
print_class_distribution(w)

