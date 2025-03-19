#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2023-07-22
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/scikit-radar
# @File    : pcd.py
# @IDE     : vscode

"""Visualization Tools of Point Cloud."""

from typing import Any, Optional, Union, Tuple

import cv2
import numpy as np
import open3d as o3d
from .visual_geometry import camera_coordinate_to_uvd


def open3d_pointcloud_to_numpy(pcd: o3d.geometry.PointCloud) -> np.ndarray:
    """
    Converts an Open3D point cloud to a NumPy array.

    Args:
        pcd (open3d.geometry.PointCloud): Open3D point cloud object.

    Returns:
        numpy.ndarray: NumPy array representing the point cloud with shape (N, 3).

    Raises:
        ValueError: If the input object is not an Open3D point cloud.
    """

    if isinstance(pcd, o3d.geometry.PointCloud):
        pcd_array = np.asarray(pcd.points)
    else:
        raise ValueError("Input object should be an open3d.geometry.PointCloud object")

    return pcd_array


def numpy_to_open3d_pointcloud(points: np.ndarray, colors: Optional[np.ndarray] = None) -> o3d.geometry.PointCloud:
    """
    Converts a NumPy array to Open3D point cloud data.

    Args:
        numpy_array (numpy.ndarray): NumPy array representing the point cloud with shape (N, 3).

    Returns:
        open3d.geometry.PointCloud: Open3D point cloud data.

    Raises:
        ValueError: If the input object is not a NumPy array.
    """

    if isinstance(points, np.ndarray):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
    else:
        raise ValueError("Input object should be a numpy.ndarray object")

    if colors is not None and len(colors) != 0:
        if np.max(colors) > 1:
            colors = (colors / np.max(colors)).astype(np.float32)
        # make sure numpy version < 2.0.0 (not fixed until 2025-02-27)
        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def load_point_cloud(pcd_path: str) -> o3d.geometry.PointCloud:
    """
    Load a point cloud from a file.

    Args:
        pcd_path (str): Path to the point cloud file.

    Returns:
        o3d.geometry.PointCloud: Loaded point cloud.
    """
    return o3d.io.read_point_cloud(pcd_path)


def save_point_cloud(pcd: o3d.geometry.PointCloud, save_path: str) -> None:
    """
    Save a point cloud to a file.

    Args:
        pcd (o3d.geometry.PointCloud): Point cloud to be saved.
        save_path (str): Path to save the point cloud.
    """
    o3d.io.write_point_cloud(save_path, pcd)


def visualize_point_cloud(pcd: Union[o3d.geometry.PointCloud, list], title="point cloud"):
    """
    Visualize one or more point clouds.

    Args:
        pcd (Union[o3d.geometry.PointCloud, list]): Point cloud or a list of point clouds.
        title (str): Title for the visualization window.
    """
    # Visualize point cloud
    if isinstance(pcd, list):
        pass  # Placeholder for handling multiple point clouds
    else:
        pcd = [pcd]

    o3d.visualization.draw_geometries(pcd, window_name=title)


def draw_aabb(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """
    Draw an Axis-Aligned Bounding Box (AABB) around a point cloud.

    Args:
        pcd (o3d.geometry.PointCloud): Input point cloud.

    Returns:
        o3d.geometry.PointCloud: Point cloud representing the AABB.
    """
    aabb = pcd.get_axis_aligned_bounding_box()
    aabb.color = (1, 0, 0)
    return aabb


def draw_obb(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """
    Draw an Oriented Bounding Box (OBB) around a point cloud.

    Args:
        pcd (o3d.geometry.PointCloud): Input point cloud.

    Returns:
        o3d.geometry.PointCloud: Point cloud representing the OBB.
    """
    bbx = pcd.get_oriented_bounding_box()
    bbx.color = (0, 0, 1)
    return bbx

def draw_xyz_frame() -> o3d.geometry.TriangleMesh:
    """
    Creates and returns a 3D coordinate frame mesh using Open3D.

    Returns:
    - o3d.geometry.TriangleMesh: A 3D mesh representing the XYZ coordinate frame.
        The x, y, z axis will be rendered as red, green, and blue arrows respectively.
    """
    # Create coordinate frame using Open3D
    return o3d.geometry.TriangleMesh.create_coordinate_frame()


def get_object_aabb_corners(
    aabb: o3d.geometry.PointCloud, as_array: bool = True
) -> Union[o3d.geometry.PointCloud, np.ndarray]:
    """
    Get the 8 corner coordinates of an Axis-Aligned Bounding Box (AABB).

    Args:
        aabb (o3d.geometry.PointCloud): Point cloud representing the AABB.
        as_array (bool): If True, return corners as a NumPy array.

    Returns:
        Union[o3d.geometry.PointCloud, np.ndarray]: Corner coordinates of the AABB.
    """
    # Get 8 corner coordinates of AABB
    corners = aabb.get_box_points()

    if as_array:
        corners = open3d_pointcloud_to_numpy(corners)

    return corners


def merge_point_clouds(point_clouds: list) -> o3d.geometry.PointCloud:
    """
    Merge multiple point clouds into a single point cloud.

    Args:
        point_clouds (list): List of point clouds to merge.

    Returns:
        o3d.geometry.PointCloud: Merged point cloud.
    """
    # Concatenate point coordinates and colors
    merged_points = np.concatenate([np.asarray(pcd.points) for pcd in point_clouds], axis=0)
    merged_colors = np.concatenate([np.asarray(pcd.colors) for pcd in point_clouds], axis=0)

    # Create a new point cloud

    return numpy_to_open3d_pointcloud(points=merged_points, colors=merged_colors)


def sample_point_cloud(
    input_pcd: o3d.geometry.PointCloud, sample_mask: Optional[np.ndarray] = None, sample_indices: Union[list, np.ndarray] = None
) -> o3d.geometry.PointCloud:
    """
    Sample a point cloud based on a mask or index and return a new point cloud with colors.

    Parameters:
    - input_pc: Input Open3D point cloud.
    - sample_mask: A boolean mask indicating which points to sample.
    - sample_indices: A list of indices indicating which points to sample.

    Returns:
    - sampled_pc: Sampled point cloud with colors.
    """
    if sample_mask is not None and sample_indices is not None:
        raise ValueError("Please provide either sample_mask or sample_indices")

    points = np.asarray(input_pcd.points)
    colors = np.asarray(input_pcd.colors)

    if sample_mask is not None:
        if len(sample_mask) != len(points):
            raise ValueError("Length of sample_mask must be the same as the number of points in the input point cloud.")
        sampled_indices = np.where(sample_mask)[0]
    elif sample_indices is not None:
        sampled_indices = sample_indices
    else:
        raise ValueError("Please provide either sample_mask or sample_indices.")

    sampled_points = points[sampled_indices]
    sampled_colors = colors[sampled_indices]

    return numpy_to_open3d_pointcloud(points=sampled_points, colors=sampled_colors)


def pick_points_from_pcd(pcd: Union[list, o3d.geometry.PointCloud]) -> list:
    """
    Allow the user to pick points from a 3D point cloud.

    Args:
        pcd (Union[list, o3d.geometry.PointCloud]): Input point cloud or a list of point clouds.

    Returns:
        list: List of indices of picked points.
    """
    if isinstance(pcd, list):
        pcd = merge_point_clouds(pcd)

    print("")
    print("1) Please pick at least three correspondences using [shift + left click]")
    print("   Press [shift + right click] to undo point picking")
    print("2) After picking points, press 'Q' to close the window")

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # User picks points
    vis.destroy_window()
    print("")

    return vis.get_picked_points()


def shift_point_cloud_to_origin(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """
    Shift the center of a point cloud to the origin (0, 0, 0).

    Args:
        pcd (o3d.geometry.PointCloud): Input point cloud.

    Returns:
        o3d.geometry.PointCloud: Point cloud with the center shifted to the origin.
    """
    # Get the centroid of the point cloud
    centroid = np.asarray(pcd.get_center())

    # Compute the translation vector to shift the centroid to the origin
    translation_vector = -centroid

    # Translate the point cloud
    pcd_shifted = pcd.translate(translation_vector)

    return pcd_shifted


def draw_3d_bounding_box(
    image: np.ndarray, corners: np.ndarray, intrinsic: np.ndarray, object_label: Optional[str] = None, color: tuple = (0, 1, 0)
) -> np.ndarray:
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1

    Draw a 3D bounding box on the given image.

    :param image: The RGB image as a numpy array.
    :param corners: An array of corner points. Shape should be (8, 3).
    :return: Image with the 3D bounding box drawn on it.
    """
    # project camera coordinate of corners to uv-coordinate
    # id any point depth is negative, draw nothing and return original image
    corners = camera_coordinate_to_uvd(corners, intrinsic, as_tuple=False)
    if np.any(corners[:, 2] < 0):
        return image

    # Original image dimensions
    original_height, original_width = image.shape[:2]

    # Find the padding required
    padding_x = int(max(-corners[:, 0].min(), corners[:, 0].max() - original_width, 0))
    padding_y = int(max(-corners[:, 1].min(), corners[:, 1].max() - original_height, 0))

    # Create a larger canvas (padded image)
    try:
        padded_image = np.zeros((original_height + 2 * padding_y, original_width + 2 * padding_x, 3), dtype=image.dtype)
        padded_image[padding_y : padding_y + original_height, padding_x : padding_x + original_width] = image

        # Adjust corners to new canvas
        corners[:, 0] += padding_x
        corners[:, 1] += padding_y

        # Define connections between corners of the bounding box
        lines = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Lower square
            (4, 5), (5, 6), (6, 7), (7, 4),  # Upper square
            (0, 4), (1, 5), (2, 6), (3, 7),  # Vertical lines
            (0, 7), (3, 4),
        ]

        # Draw lines on the padded image
        for start, end in lines:
            start_point = tuple(corners[start, :2].astype(int))
            end_point = tuple(corners[end, :2].astype(int))
            cv2.line(padded_image, start_point, end_point, color, 1, lineType=cv2.LINE_AA)

        # Draw object label above the bounding box
        if object_label is not None:
            label_position = (int(corners.mean(axis=0)[0]), int(corners[:, 1].min()) - 5)
            cv2.putText(padded_image, object_label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Number the corners
        # for idx, corner in enumerate(corners):
        #     corner = tuple(corner[:2].astype(int))
        #     cv2.putText(padded_image, str(idx), corner, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 1), 1)

        # Crop back to original image dimensions
        cropped_image = padded_image[padding_y : padding_y + original_height, padding_x : padding_x + original_width]
    except:
        cropped_image = np.copy(image)
    return cropped_image


def down_sample_point_cloud(input_pcd: o3d.geometry.PointCloud, total_num_sample: int) -> np.ndarray:
    """
    Sample the input Open3D point cloud by dividing it into a 3D grid and uniformly sampling a specified number of points.

    Parameters:
    - input_pcd: Input Open3D point cloud object.
    - total_num_sample: Total number of points to be sampled.

    Returns:
    - sampled_indices: Sampled Open3D point cloud object.
    """

    # Calculate the grid resolution based on the total number of points
    total_num_points = len(input_pcd.points)
    sample_grid_resolution = (total_num_points / total_num_sample) ** (1 / 3)

    # Calculate the grid index for each point
    grid_indices = np.floor(np.asarray(input_pcd.points) / sample_grid_resolution).astype(int)

    # Use a dictionary to store the indices of points in each grid voxel
    grid_points_dict = {}
    for i, grid_index in enumerate(grid_indices):
        grid_index_tuple = tuple(grid_index)
        if grid_index_tuple not in grid_points_dict:
            grid_points_dict[grid_index_tuple] = []
        grid_points_dict[grid_index_tuple].append(i)

    # Calculate the number of points to be sampled in each grid voxel
    total_num_grid = len(grid_points_dict)
    sample_count_per_grid = int(total_num_sample / total_num_grid)

    # Uniformly sample points within each grid voxel
    sampled_indices = []
    for grid_index_tuple, point_indices in grid_points_dict.items():
        if len(point_indices) >= sample_count_per_grid:
            sampled_indices.extend(np.random.choice(point_indices, sample_count_per_grid, replace=False))
        else:
            sampled_indices.extend(point_indices)

    return np.asarray(sampled_indices)


def random_sample_point_cloud(input_pcd: o3d.geometry.PointCloud, total_num_sample: int) -> np.ndarray:
    """
    Randomly sample a specified number of points from a 3D point cloud.

    Args:
        input_pcd (o3d.geometry.PointCloud): Input point cloud.
        num_samples (int): Number of points to randomly sample.

    Returns:
        o3d.geometry.PointCloud: Point cloud with randomly sampled points.
    """

    # Generate random indices for sampling without replacement
    return np.random.choice(len(np.asarray(input_pcd.points)), size=total_num_sample, replace=False)


from .visual_geometry import *

def rgbd_to_point_cloud(rgb_image: np.ndarray, depth_image: np.ndarray, intrinsic: np.ndarray) -> o3d.geometry.PointCloud:
    """
    Convert RGB-D images to colored point cloud.

    Parameters:
        rgb_image (np.ndarray): The RGB image as a numpy array of shape (height, width, 3).
        depth_image (np.ndarray): The depth image as a numpy array of shape (height, width).
        intrinsic (np.ndarray): The camera intrinsic parameters as a numpy array of shape (3, 3).

    Returns:
        open3d.geometry.PointCloud: A colored point cloud object.
    """
    # Convert depth image to camera coordinate
    points_3d = depth_image_to_camera_coordinate(depth_image, intrinsic)
    
    # Resize RGB image to match depth image size
    rgb_image = cv2.resize(rgb_image, dsize=(640, 480))
    
    # Flatten RGB image to get colors for each point
    colors = rgb_image.reshape((-1, 3))
    
    # Filter out points with zero depth
    colors = colors[depth_image.reshape(-1) != 0]
    
    # Normalize colors if necessary
    if np.max(colors) > 1:
        colors = (colors / np.max(colors)).astype(np.float32)

    # Create colored point cloud
    colored_pcd = numpy_to_open3d_pointcloud(points=points_3d, colors=colors)
    return colored_pcd

def point_cloud_to_rgbd(pcd: o3d.geometry.PointCloud, camera_pose: np.ndarray, intrinsic: np.ndarray, view_range: Tuple[int, int] = (192, 256)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert point cloud to RGB-D images.

    Parameters:
        pcd (open3d.geometry.PointCloud): The input colored point cloud.
        camera_pose (np.ndarray): The camera pose as a numpy array of shape (4, 4).
        intrinsic (np.ndarray): The camera intrinsic parameters as a numpy array of shape (3, 3).
        view_range (Tuple[int, int], optional): The dimensions of the output RGB-D images. Defaults to (192, 256).

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the RGB image and the depth image as numpy arrays.
    """
    # Extract points and colors from point cloud
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # Convert world coordinates to camera coordinates
    cc_pts = world_coordinate_to_camera_coordinate(points, camera_pose)

    # Project camera coordinates to UV depth
    scaled_u, scaled_v, depth = camera_coordinate_to_uvd(cc_pts, intrinsic)

    # Round and convert UV coordinates to integers
    u = np.round(scaled_u).astype(int)
    v = np.round(scaled_v).astype(int)

    # Define FOV dimensions
    row, col = view_range

    # Find valid indices within image range
    valid_idx = (u >= 0) & (v >= 0) & (u < col) & (v < row)

    # Retrieve valid pixel coordinates and depths
    u_valid = u[valid_idx]
    v_valid = v[valid_idx]
    depth_valid = depth[valid_idx]
    color_valid = colors[valid_idx]

    # Sort depth values from large to small
    depth_sort_idx = np.argsort(depth_valid)[::-1]

    # Initialize depth and RGB images
    depth_image = np.zeros(view_range, dtype=np.float32)
    rgb_image = np.ones((*view_range, 3), dtype=np.float32)

    # Populate depth and RGB images with valid data
    depth_image[v_valid[depth_sort_idx], u_valid[depth_sort_idx]] = depth_valid[depth_sort_idx]
    rgb_image[v_valid[depth_sort_idx], u_valid[depth_sort_idx]] = color_valid[depth_sort_idx, :]

    return rgb_image, depth_image







from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

def filter_segmented_point_cloud(pcd, eps=0.02, min_samples=50):
    # Convert Open3D point cloud to numpy array
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    # Scale the point cloud features
    scaler = StandardScaler()
    points_scaled = scaler.fit_transform(points)
    
    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(points_scaled)
    
    # Find the largest cluster
    largest_cluster_label = np.argmax(np.bincount(labels[labels != -1]))
    
    # Extract points belonging to the largest cluster
    largest_cluster_indices = np.where(labels == largest_cluster_label)[0]
    largest_cluster_points = points[largest_cluster_indices]
    colors = colors[largest_cluster_indices]
    
    # Create an Open3D point cloud for the largest cluster
    largest_cluster_pcd = o3d.geometry.PointCloud()
    largest_cluster_pcd.points = o3d.utility.Vector3dVector(largest_cluster_points)
    largest_cluster_pcd.colors = o3d.utility.Vector3dVector(colors)

    
    return largest_cluster_pcd

def icp_registration(source_points, target_points, max_correspondence_distance=0.00001, max_iterations=50):
    """
    Perform ICP registration between two point clouds.
    
    source_points: numpy array of shape (N, 3)
        Source point cloud.
    target_points: numpy array of shape (M, 3)
        Target point cloud.
    max_correspondence_distance: float, optional
        Maximum correspondence distance to consider a point pair as a match (default is 0.05).
    max_iterations: int, optional
        Maximum number of iterations for ICP (default is 50).
        
    Returns:
        transformation: numpy array of shape (4, 4)
            Transformation matrix that aligns the source point cloud to the target point cloud.
    """
    source_cloud = o3d.geometry.PointCloud()
    source_cloud.points = o3d.utility.Vector3dVector(source_points)

    target_cloud = o3d.geometry.PointCloud()
    target_cloud.points = o3d.utility.Vector3dVector(target_points)

    # Perform ICP registration
    icp_result = o3d.pipelines.registration.registration_icp(
        source_cloud, target_cloud, max_correspondence_distance, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)
    )

    return icp_result.transformation