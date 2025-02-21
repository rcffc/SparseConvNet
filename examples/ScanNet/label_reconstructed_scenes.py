from utils.write_ply import write_ply
import open3d as o3d
import numpy as np
import copy
from sklearn.neighbors import KDTree, KNeighborsClassifier, NearestNeighbors
import plyfile
import glob
import torch
import os

k = 3

# Map relevant classes to {0,1,...,19}, and ignored classes to -100
remapper=np.ones(150)*(-100)
for i,x in enumerate([1,2,3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39]):
    remapper[x]=i


def draw_registration_result(source, target, transformation, scene=None):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)

    # o3d.visualization.draw_geometries([source_temp, target_temp],
    #                                   zoom=0.4559,
    #                                   front=[0.6452, -0.3036, -0.7011],
    #                                   lookat=[1.9892, 2.0208, 1.8945],
    #                                   up=[-0.2779, -0.9482, 0.1556])
    # if scene:
    #     vis = o3d.visualization.Visualizer()
    #     vis.create_window(visible=False)
    #     vis.add_geometry(source_temp)
    #     vis.add_geometry(target_temp)
    #     # vis.poll_events()
    #     # vis.update_renderer()
    #     vis.capture_screen_image(CAPTURES_DIR.format(scene))
    #     vis.destroy_window()


def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result


def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size, result_ransac):
    source.estimate_normals()
    target.estimate_normals()
    distance_threshold = voxel_size * 0.4
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result


def do_registration(source, target, scene):
    voxel_size = 0.05
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

    result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)
    # print(result_ransac)

    # draw_registration_result(source_down, target_down, result_ransac.transformation, scene)

    result_icp = refine_registration(source, target, source_fpfh, target_fpfh,
                                     voxel_size, result_ransac)
    # print(result_icp)
    return result_icp


def label_reconstruction(reconstruction, target, target_path):
    # read ground truth labels:
    target_ply = plyfile.PlyData().read(target_path)
    gt_labels = remapper[np.array(target_ply.elements[0]['label'])]

    clf = KNeighborsClassifier(k, weights='distance')
    clf.fit(np.asarray(target.points), gt_labels)

    return clf.predict(np.array(reconstruction.points))

train_or_val = 'train'
source_paths = sorted(glob.glob('/opt/datasets/'+ train_or_val + '/scenes/multi/pynfinitam/*.pth'))
prefix_len = len('/opt/datasets/'+ train_or_val + '/scenes/multi/pynfinitam/')
scene_id_len = len('0444_00')
SAVE_SCENE_MASK_PTH = '/opt/datasets/'+ train_or_val + '/scenes/multi/labeled/{}.pth'
SAVE_SCENE_MASK_PLY = '/opt/datasets/'+ train_or_val + '/scenes/multi/labeled/{}.ply'
# SAVE_SCENE_MASK_PTH_REVIEW = '/opt/datasets/val/scenes/multi/labeled/{}_REVIEW.pth'
# SAVE_SCENE_MASK_PLY_REVIEW = '/opt/datasets/val/scenes/multi/labeled/{}_REVIEW.ply'
# CAPTURES_DIR = '/opt/datasets/val/scenes/multi/captures/{}.png'

for source_path in source_paths:
    scene_id = source_path[prefix_len:-4]
    
    if os.path.exists(SAVE_SCENE_MASK_PTH.format(scene_id)):
        print(scene_id + ' already labeled!')
        continue
    print(scene_id)
    print(source_path)
    target_path = '/opt/datasets/scans/scene{}/scene{}_vh_clean_2.labels.ply'.format(scene_id[:scene_id_len], scene_id[:scene_id_len])

    source_data = torch.load(source_path).astype(np.int32)
    source = o3d.geometry.PointCloud()
    try:
        source.points = o3d.utility.Vector3dVector(source_data[:,:3])
        source.colors = o3d.utility.Vector3dVector(source_data[:,3:6])

        target = o3d.io.read_point_cloud(target_path)
        target_mesh = o3d.io.read_triangle_mesh(target_path)

        source.scale(0.005, np.zeros(3))

        result = do_registration(source, target, scene_id)
        source.transform(result.transformation)

        labeled_reconstruction = label_reconstruction(source, target, target_path)

        # Save die labels im label feld aber original farben bleiben erhalten
        torch.save((np.asarray(source.points), np.asarray(source.colors), labeled_reconstruction), SAVE_SCENE_MASK_PTH.format(scene_id))
        # Save pointcloud with colors as labels
        write_ply(labeled_reconstruction, (np.asarray(source.points)), SAVE_SCENE_MASK_PLY.format(scene_id))

    except IndexError as e:
        print (e)
    