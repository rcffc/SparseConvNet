import open3d as o3d
import numpy as np
import copy
from sklearn.neighbors import KDTree, KNeighborsClassifier, NearestNeighbors
import plyfile
import glob
import torch
import iou, accuracy
import time
import multiprocessing as mp

k = 3


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
    #     vis.create_window()
    #     vis.add_geometry(source_temp)
    #     vis.add_geometry(target_temp)
    #     vis.poll_events()
    #     vis.update_renderer()
    #     vis.capture_screen_image('captures/{}.png'.format(scene))
    #     vis.destroy_window()


def preprocess_point_cloud(pcd, voxel_size):
    # print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    # print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    # print(":: RANSAC registration on downsampled point clouds.")
    # print("   Since the downsampling voxel size is %.3f," % voxel_size)
    # print("   we use a liberal distance threshold %.3f." % distance_threshold)
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
    # print(":: Point-to-plane ICP registration is applied on original point")
    # print("   clouds to refine the alignment. This time we use a strict")
    # print("   distance threshold %.3f." % distance_threshold)
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
    # draw_registration_result(source, target, result_icp.transformation)
    return result_icp.transformation


def get_knn_pc(source, target):
    source_pts = np.asarray(source.points)
    target_pts = np.asarray(target.points)
    tree = KDTree(source_pts)
    knn_pc = []  # np.empty(shape=(len(source_pts), k))
    for i, point in enumerate(target_pts):
        indices = tree.query(np.array([point]), k=k, return_distance=False)[0]
        knn_pc.append(source_pts[indices])
    return np.array(knn_pc)

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def calculate_distance_point_to_triangle(p, p0, p1, p2):
    n = np.cross(p2-p0, p2-p1)
    n = normalize(n)
    dist = np.abs(np.dot(p-p0,n))
    return dist

def get_triangles(id, gt):
    return np.where(gt[:,]==id)[0]

distance_threshold = 0.02

files = glob.glob('/opt/datasets/reconstructed/*.ply')
failed_train = open('/app/src/examples/ScanNet/failed_scenes_train.txt', 'r')
failed_val = open('/app/src/examples/ScanNet/failed_scenes_val.txt', 'r')
failed_lines = [s.strip() for s in failed_train.readlines()]
failed_lines.append([s.strip() for s in failed_val.readlines()])
files = [f for f in files if f[len('/opt/datasets/reconstructed/'):-len('.ply')] not in failed_lines]

def calculate_reconstruction_accuracy(src, gt):
    distances = []
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(np.asarray(gt.vertices))
    for y, vtx in enumerate(src):
        # the indices of the three closest vertices
        dists, indices = nbrs.kneighbors([vtx], return_distance=True)
        triangles = []
        for x, id in enumerate(indices[0]):
            if dists[0][x] > distance_threshold:
                continue
            for triangle in get_triangles(id, np.asarray(gt.triangles)):
                # add vertices of triangle
                triangles.append(np.asarray(gt.triangles)[triangle])
        

        # get all triangles that includes these points
        distance = np.Inf
        for p0, p1, p2 in triangles:
            new_distance = calculate_distance_point_to_triangle(vtx, np.asarray(gt.vertices)[p0], np.asarray(gt.vertices)[p1], np.asarray(gt.vertices)[p2])
            if new_distance < distance:
                distance = new_distance
        
        # append lowest distance
        distances.append(distance)
    distances[:] = [x if x != np.Inf else np.nan for x in distances]
    return np.nanmean(np.array(distances))

def evaluate_reconstruction(source, target_mesh, scene_id):
    print(source)
    start=time.time()
    mean_reconstruction_accuracy = calculate_reconstruction_accuracy(np.asarray(source.points), target_mesh)
    print(scene_id, '\t', mean_reconstruction_accuracy, '\t', time.time()-start)
    return mean_reconstruction_accuracy

def evaluate_segmentation(source, target, source_path, scene_id):
    # read grount truth labels:
    # target_pth_path = '/opt/datasets/scannetv2_sparseconvnet/train/scene0444_00_vh_clean_2.pth'
    target_pth_path = '/igd/a4/homestud/pejiang/ScanNet/scans/scene{}/scene{}_vh_clean_2.labels.pth'.format(scene_id, scene_id)
    target_pth = glob.glob(target_pth_path)[0]
    _, _, gt_labels = torch.load(target_pth)

    # read source labels
    a = plyfile.PlyData().read(source_path)
    predicted_labels_reconstructed = np.array(a.elements[0]['label'])

    clf = KNeighborsClassifier(k, weights='distance')
    clf.fit(np.asarray(source.points), predicted_labels_reconstructed)
    predicted_labels_original_pc = clf.predict(np.array(target.points))

    iou.evaluate(predicted_labels_original_pc, gt_labels.astype(int))
    accuracy.evaluate(predicted_labels_original_pc, gt_labels.astype(int))


source_paths = glob.glob('/opt/datasets/reconstructed/*.ply')
prefix_len = len('/opt/datasets/reconstructed/')

MRA = []

def evaluate_rec(source_path):
    if source_path not in files:
        print('Skip ' + source_path + ', low quality reconstruction')
        return

    scene_id = source_path[prefix_len:-len('.ply')]    
    with open('/app/results/nohup.out', 'r') as results:
        results_lines = results.readlines()
        for line in results_lines:
            if line.startswith(scene_id[:7]):
                print('Skip ' + scene_id + ', already evaluated')
                return

    target_path = '/opt/datasets/scans/scene{}/scene{}_vh_clean_2.labels.ply'.format(scene_id[:7], scene_id[:7])

    source = o3d.io.read_point_cloud(source_path)
    target = o3d.io.read_point_cloud(target_path)
    target_mesh = o3d.io.read_triangle_mesh(target_path)

    transformation = do_registration(source, target, scene_id)
    source.transform(transformation)
    MRA.append(evaluate_reconstruction(source, target_mesh, scene_id))

for source in source_paths:
    evaluate_rec(source)  

print(MRA)
MRA_average = sum(MRA)/len(MRA)
print(MRA_average)