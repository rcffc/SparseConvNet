import numpy as np
import plyfile

color_dict={}
color_dict[0] = np.array([174, 199, 232])
color_dict[1] = np.array([152, 223, 138])
color_dict[2] = np.array([31, 119, 180])
color_dict[3] = np.array([255, 187, 120])
color_dict[4] = np.array([188, 189,  34])
color_dict[5] = np.array([140,  86,  75])
color_dict[6] = np.array([255, 152, 150])
color_dict[7] = np.array([214,  39,  40])
color_dict[8] = np.array([197, 176, 213])
color_dict[9] = np.array([148, 103, 189])
color_dict[10] = np.array([196, 156, 148])
color_dict[11] = np.array([23, 190, 207])
color_dict[12] = np.array([247, 182, 210])
color_dict[13] = np.array([219, 219, 141])
color_dict[14] = np.array([255, 127,  14])
color_dict[15] = np.array([158, 218, 229])
color_dict[16] = np.array([44, 160,  44])
color_dict[17] = np.array([112, 128, 144])
color_dict[18] = np.array([227, 119, 194])
color_dict[19] = np.array([82,  84, 163])
color_dict[-100] = np.array([0,  0, 0])

def write_ply(labels, positions, path):
    global color_dict    
    colors = np.array([color_dict.__getitem__(label) for label in labels])

    vertices = np.empty(len(labels), dtype=[(
        'x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('label', 'i4')])
    vertices['x'] = positions[:, 0].astype('f4')
    vertices['y'] = positions[:, 1].astype('f4')
    vertices['z'] = positions[:, 2].astype('f4')
    vertices['red'] = colors[:, 0].astype('u1')
    vertices['green'] = colors[:, 1].astype('u1')
    vertices['blue'] = colors[:, 2].astype('u1')
    vertices['label'] = labels.astype('i4')
    
    ply = plyfile.PlyData(
        [plyfile.PlyElement.describe(vertices, 'vertex')], text=False)
    ply.write(path)
    print('Finished writing')
