#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2023-07-22
# @Author  : Zhaoze Wang
# @Site    : https://github.com/Wangzhaoze/pyradar
# @File    : visual_geometry.py
# @IDE     : vscode

"""Tools of Visual Geometry."""

from scipy.spatial.transform import Rotation
from typing import Any, Union

import numpy as np
import open3d as o3d
import torch


def is_xyz_coordinate(input_data: Any) -> bool:
    """
    Check if the input data represents a valid set of 3D points.

    Args:
        input_data (Any): Input data to be checked.

    Returns:
        bool: True if the input data is a valid set of 3D points, False otherwise.

    Raises:
        ValueError: If the input data is not a numpy.ndarray or does not have the shape (N, 3).

    Note:
        A valid set of 3D points should be represented as a numpy array with shape (N, 3), where N is the number
        of points, and each point is represented by a 3-dimensional vector (x, y, z). This function checks if
        the input data satisfies these conditions and returns True if it is a valid set of 3D points, and raises
        a ValueError with an appropriate error message if it is not.

    Example of a valid set of 3D points:
        [[x1, y1, z1],
         [x2, y2, z2],
         ...,
         [xN, yN, zN]]

    """
    if isinstance(input_data, np.ndarray) and input_data.shape[1] == 3:
        return True
    else:
        raise ValueError(
            'Input data should be a numpy.ndarray and shape should be (N, 3).'
        )


def is_homogeneous_coordinate(input_data: Any) -> bool:
    """
    Check if a given matrix is a homogeneous matrix.

    Args:
        input_data (Any): The input matrix to be checked.

    Returns:
        bool: True if the matrix is a homogeneous matrix, False otherwise.

    Raises:
        ValueError: If input_data is not a numpy.ndarray.
        ValueError: If the input matrix is not square.
        ValueError: If the last row is not [0, 0, ..., 0, 1].
    """

    if not isinstance(input_data, np.ndarray):
        raise ValueError('input data should be numpy.ndarray')

    # Check if the matrix is square
    if input_data.shape[1] != 4:
        raise ValueError('The input matrix should be square')

    # Check if the last colume is not all ones
    if input_data[:, -1].any() != 1:
        raise ValueError('The last row should be [0, 0, ..., 0, 1]')

    return True


def is_intrinsic_matrix(input_data: Any) -> bool:
    """
    Check if the input data is a valid camera intrinsic matrix.

    Args:
        input_data (Any): Input data to be checked.

    Returns:
        bool: True if the input data is a valid camera intrinsic matrix, False otherwise.

    Raises:
        ValueError: If the input data is not a numpy.ndarray or does not have the shape (3, 3).
        ValueError: If any of the focal lengths (fx and fy) are zero.
        ValueError: If any of the principal point coordinates (cx and cy) are zero.
        ValueError: If the right down value of the intrinsic matrix is not one.
        ValueError: If any other elements in the intrinsic matrix are non-zero.

    Note:
        A camera intrinsic matrix should have the shape (3, 3). Additionally, the focal lengths (fx and fy) should
        not be zero, and the principal point coordinates (cx and cy) should not be zero. The right down value of
        the intrinsic matrix should be one, and all other elements should be zeros. This function checks if the
        input data satisfies these conditions and returns True if it is a valid intrinsic matrix, and raises a
        ValueError with an appropriate error message if it is not.

    Example of a valid camera intrinsic matrix:
            [[fx   0  cx],
            [ 0  fy  cy],
            [ 0   0   1]]
    """

    if not isinstance(input_data, np.ndarray):
        raise ValueError('Input data should be a numpy.ndarray')

    if input_data.shape != (3, 3):
        raise ValueError('Input data shape should be (3, 3)')

    # Check if any of the focal lengths fx and fy are zero
    if input_data[0, 0] == 0 or input_data[1, 1] == 0:
        raise ValueError('Focal lengths fx and fy should not be zero.')

    # Check if any of the principal point coordinates cx and cy are zero
    if input_data[0, 2] == 0 or input_data[1, 2] == 0:
        raise ValueError('Principal point coordinates cx and cy should not be zero.')

    # Check if the right down value of the intrinsic matrix is one
    if input_data[2, 2] != 1:
        raise ValueError('Right down value of the intrinsic matrix should be one.')

    # Check if any other elements in the intrinsic matrix are non-zero
    if (
        np.any(input_data[0:2, 0:2] != np.diag(np.diagonal(input_data[0:2, 0:2])))
        or input_data[2, 0] != 0
        or input_data[2, 1] != 0
    ):
        raise ValueError('Other elements in the intrinsic matrix should be zeros.')

    # If all the checks pass, then the input_data represents a valid intrinsic
    # matrix
    return True


def is_transformation_matrix(input_data: np.ndarray) -> bool:
    """
    Check if the input data is a valid 4x4 transformation matrix.

    Parameters:
        input_data (numpy.ndarray): The input data to be checked.

    Returns:
        bool: True if the input data is a valid transformation matrix, False otherwise.
    """
    ERROR_TOLERANCE: float = 1e-6

    # Check if the input data is a numpy array
    if not isinstance(input_data, np.ndarray):
        raise ValueError('input data should be numpy.ndarray')

    # Check if the matrix is 4x4
    if input_data.shape != (4, 4):
        raise ValueError('shape of pose matrix should be (4, 4)')

    # Check if the upper-left 3x3 sub-matrix is a unitary rotation matrix
    rotation_matrix = input_data[:3, :3]
    if not np.allclose(
        np.dot(rotation_matrix.T, rotation_matrix), np.eye(3), atol=ERROR_TOLERANCE
    ):
        raise ValueError('Rotation Matrix should be identity')

    # Check if the last row is [0, 0, 0, 1]
    last_row = input_data[3]
    if not np.allclose(last_row, [0, 0, 0, 1], atol=ERROR_TOLERANCE):
        raise ValueError('The last row should be [0, 0, 0, 1]')

    return True


def is_pose_matrix(input_data: Any) -> bool:
    """
    Check if a given matrix is a 4x4 pose matrix representing rotation and translation.

    Args:
        input_data (Any): The input matrix to be checked.

    Returns:
        bool: True if the matrix is a 4x4 pose matrix, False otherwise.

    Raises:
        ValueError: If input_data is not a numpy.ndarray.
        ValueError: If the shape of the pose matrix is not (4, 4).
        ValueError: If the upper-left 3x3 sub-matrix is not a unitary rotation matrix.
        ValueError: If the last row is not [0, 0, 0, 1].
    """

    ERROR_TOLERANCE: float = 1e-4

    if not isinstance(input_data, np.ndarray):
        raise ValueError('input data should be numpy.ndarray')

    # Check if the matrix is 4x4
    if input_data.shape != (4, 4):
        raise ValueError('shape of pose matrix should be (4, 4)')

    # Check if the upper-left 3x3 sub-matrix is a unitary rotation matrix
    rotation_matrix = input_data[:3, :3]
    if not np.allclose(
        np.dot(rotation_matrix.T, rotation_matrix), np.eye(3), atol=ERROR_TOLERANCE
    ):
        raise ValueError('Rotation Matrix should be identity')

    # Check if the last row is [0, 0, 0, 1]
    last_row = input_data[3]
    if not np.allclose(last_row, [0, 0, 0, 1], atol=ERROR_TOLERANCE):
        raise ValueError('The last row should be [0, 0, 0, 1]')

    return True


def inverse_pose(pose: np.ndarray) -> np.ndarray:
    """
    Compute the inverse of a 4x4 pose matrix.

    Args:
        pose (np.ndarray): Input pose matrix with shape (4, 4).

    Returns:
        np.ndarray: Inverse of the input pose matrix.

    Raises:
        ValueError: If the input pose matrix is not in the correct format.
        ValueError: If the input pose matrix is not invertible.

    Note:
        This function takes a 4x4 pose matrix as input and computes its inverse.
        The input pose matrix should be in the format [R | t], where R is a 3x3 rotation matrix,
        and t is a 3x1 translation vector.

        The function performs input validation to check if the input pose is in the correct format.
        If the input does not meet the required format, a ValueError is raised with the appropriate error message.

        Additionally, the function checks if the input pose matrix is invertible.
        If the input pose matrix is not invertible, a ValueError is raised with the appropriate error message.
    """

    if not is_pose_matrix(pose):
        raise ValueError('The input pose matrix should be in correct format')

    # Extract the rotation matrix and translation vector from the pose
    rot = pose[:3, :3]
    trans = pose[:3, 3]

    # Compute the inverse of the rotation matrix
    inv_rot = rot.T

    # Compute the inverse of the translation vector
    inv_trans = -np.dot(inv_rot, trans)

    # Construct the inverse pose matrix
    inv_pose = np.eye(4)
    inv_pose[:3, :3] = inv_rot
    inv_pose[:3, 3] = inv_trans

    if not is_pose_matrix(inv_pose):
        raise ValueError('The output pose matrix should be in correct format')

    return inv_pose


def tensor_to_array(tensor: torch.Tensor) -> np.ndarray:
    """
    Converts a PyTorch tensor to a NumPy array.

    Args:
        tensor (torch.Tensor): The input PyTorch tensor.

    Returns:
        np.ndarray: A NumPy array representing the same data as the input tensor.

    Raises:
        ValueError: If the input object is not a PyTorch tensor.
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.numpy()
    else:
        raise ValueError('Input object should be a torch.Tensor object')


def array_to_tensor(array: np.ndarray) -> torch.Tensor:
    """
    Converts a NumPy array to a PyTorch tensor.

    Args:
        array (np.ndarray): The input NumPy array.

    Returns:
        torch.Tensor: A PyTorch tensor representing the same data as the input array.

    Raises:
        ValueError: If the input object is not a NumPy array.
    """
    if isinstance(array, np.ndarray):
        return torch.from_numpy(array)
    else:
        raise ValueError('Input object should be a np.ndarray object')


def transformation_matrix_to_rotation_translation(
    transformation_matrix: np.ndarray,
) -> tuple:
    """
    Extract rotation matrix and translation vector from a 4x4 transformation_matrix matrix.

    Parameters:
        transformation_matrix (numpy.array): A 4x4 transformation_matrix matrix.

    Returns:
        Tuple(numpy.array, numpy.array): A 3x3 rotation matrix and a 3x1 translation vector.
    """
    rot = transformation_matrix[:3, :3].reshape((3, 3))
    trans = transformation_matrix[:3, 3].reshape((3, 1))
    return rot, trans


def rotation_translation_to_transformation_matrix(rot, trans):
    """
    Create a 4x4 transformation_matrix from a rotation matrix and a translation vector.

    Parameters:
        rot (numpy.array): A 3x3 rotation matrix.
        trans (numpy.array): A 3x1 translation vector.

    Returns:
        numpy.array: A 4x4 transformation_matrix representing the transformation.
    """
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rot
    transformation_matrix[:3, 3] = trans
    return transformation_matrix


def quaternion_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    """
    Convert a quaternion to a 3x3 rotation matrix.

    Parameters:
        quaternion (list or numpy.array): A list or array representing a quaternion [qw, qx, qy, qz].

    Returns:
        numpy.array: A 3x3 rotation matrix.
    """
    rotation = Rotation.from_quat(quaternion)
    rotation_matrix = rotation.as_matrix()
    return rotation_matrix


def rotation_matrix_to_quaternion(rotation_matrix):
    """
    Convert a 3x3 rotation matrix to a quaternion.

    Parameters:
        rotation_matrix (numpy.array): A 3x3 rotation matrix.

    Returns:
        list: A list representing the quaternion [qw, qx, qy, qz].
    """
    rotation = Rotation.from_matrix(rotation_matrix)
    quaternion = rotation.as_quat()
    return quaternion


def xyz_to_homogeneous(points_3d: np.ndarray) -> np.ndarray:
    """
    Convert N * 3 XYZ coordinates to N * 4 homogeneous coordinates.

    Args:
        points_3d (np.ndarray): Input XYZ coordinates with shape (N, 3).

    Returns:
        np.ndarray: Homogeneous coordinates with shape (N, 4).

    Raises:
        ValueError: If the input coordinates are not in the correct format.
        ValueError: If the output coordinates are not in the correct format.

    Note:
        This function takes N * 3 XYZ coordinates as input and directly converts them to N * 4 homogeneous coordinates
        by adding a column of 1s in the last dimension. The resulting homogeneous coordinates are returned as the output.

        The function also performs input validation to check if the input points_3d are in the correct format of XYZ coordinates.
        If the input does not meet the required format, a ValueError is raised with the appropriate error message.

        Additionally, the function checks if the output homogeneous coordinates are in the correct format. If the output does not
        meet the required format, a ValueError is raised with the appropriate error message.
    """

    if not is_xyz_coordinate(points_3d):
        raise ValueError('The input XYZ coordinates should have shape (N, 3)')

    homogeneous_coords = np.ones((points_3d.shape[0], 4))
    homogeneous_coords[:, :3] = points_3d

    if not is_homogeneous_coordinate(homogeneous_coords):
        raise ValueError(
            'The output coordinates should be in the correct format of homogeneous coordinates'
        )

    return homogeneous_coords


def homogeneous_to_xyz(homogeneous_coords: np.ndarray) -> np.ndarray:
    """
    Convert N * 4 homogeneous coordinates to N * 3 XYZ coordinates.

    Args:
        homogeneous_coords (np.ndarray): Input homogeneous coordinates with shape (N, 4).

    Returns:
        np.ndarray: XYZ coordinates with shape (N, 3).

    Raises:
        ValueError: If the input coordinates are not in the correct format.
        ValueError: If the output coordinates are not in the correct format.

    Note:
        This function takes N * 4 homogeneous coordinates as input and converts them to N * 3 XYZ coordinates
        by dividing the first three elements of the homogeneous coordinates by the last element. The resulting
        XYZ coordinates are returned as the output.

        The function also performs input validation to check if the input homogeneous_coords are in the correct format.
        If the input does not meet the required format, a ValueError is raised with the appropriate error message.

        Additionally, the function checks if the output XYZ coordinates are in the correct format. If the output does not
        meet the required format, a ValueError is raised with the appropriate error message.
    """

    if not is_homogeneous_coordinate(homogeneous_coords):
        raise ValueError('The input homogeneous coordinates should have shape (N, 4)')

    # Divide the first three elements of homogeneous coordinates by the last
    # element to get XYZ coordinates
    xyz_coords = homogeneous_coords[:, :3]

    if not is_xyz_coordinate(xyz_coords):
        raise ValueError(
            'The output coordinates should be in the correct format of XYZ coordinates'
        )

    return xyz_coords


def uv_to_flattened_idx(uv_coordinate: np.ndarray, width: int) -> np.ndarray:
    """
    Convert UV coordinates to flattened indices.

    Args:
        uv_coordinate (np.ndarray): UV coordinates, with shape (N, 2), where N is the number of coordinates.
        width (int): Width of the 2D shape. Should be positive.

    Returns:
        np.ndarray: Flattened indices, with shape (N,), corresponding to the flattened representation of UV coordinates.

    Raises:
        ValueError: If the input uv_coordinate does not have shape (N, 2).
    """
    if uv_coordinate.shape[1] != 2:
        raise ValueError('Shape of uv_coordinate should be (N, 2)')

    return uv_coordinate[:, 0] * width + uv_coordinate[:, 1]


def flattened_idx_to_uv(idx: np.ndarray, width: int) -> np.ndarray:
    """
    Convert flattened indices to UV coordinates.

    Args:
        idx (np.ndarray): Flattened indices, with shape (N,), where N is the number of indices.
        width (int): Width of the 2D shape. Should be positive.

    Returns:
        np.ndarray: UV coordinates, with shape (N, 2), corresponding to the unraveled representation of flattened indices.
    """
    u = idx // width
    v = idx % width

    return np.stack((u, v), axis=-1)


def world_coordinate_to_camera_coordinate(
    wc_points: np.ndarray, pose: np.ndarray
) -> np.ndarray:
    """
    Convert world coordinates to camera coordinates using the given pose.

    Args:
        wc_points (np.ndarray): Input 3D points in world coordinate with shape (N, 3).
        pose (np.ndarray): 4x4 pose matrix of camera representing relative rotation and translation.

    Returns:
        np.ndarray: 3D Points in camera coordinate with shape (N, 3).

    Raises:
        ValueError: If the input world coordinate points or the pose matrix does not meet the required formats.
        ValueError: If the output camera coordinate points does not meet the required formats.

    Note:
        This function takes world coordinate points and a pose matrix (4x4) representing the camera's position and orientation
        relative to the world coordinate system. It converts the points from world coordinates to camera coordinates using
        the provided pose.

        The rotation matrix (3x3) is extracted from the pose, representing the rotation from world coordinates to camera coordinates.
        The translation vector (3x1) is also extracted from the pose, representing the translation from world coordinates to camera coordinates.

        The function returns the camera coordinate points (N, 3) after the conversion.
    """

    if not is_xyz_coordinate(wc_points):
        raise ValueError('The input world coordinate points should be given in (N, 3)')

    if not is_pose_matrix(pose):
        raise ValueError('The input pose should be in correct format')

    # Extract the rotation and translation components from the pose matrix
    rot, trans = pose[:3, :3].reshape((3, 3)), pose[:3, -1].reshape((3, 1))

    # onvert world coordinates to camera coordinates using the pose : Rot @ Cc + trans = Wc
    # Here realtive R,t from pose are not equal to the usually seen R,t as in
    # formular [Xc, Yc, Zc] = R @ [Xw, Yw, Zw] + t
    cc_points = np.dot(rot.T, (wc_points.T - trans)).T

    if not is_xyz_coordinate(cc_points):
        raise ValueError(
            'The output camera coordinate points should be given in (N, 3)'
        )

    return cc_points


def camera_coordinate_to_world_coordinate(
    cc_points: np.ndarray, pose: np.ndarray
) -> np.ndarray:
    """
    Convert 3D points from camera coordinates to world coordinates using the given camera pose.

    Args:
        cc_points (np.ndarray): Input 3D Points in camera coordinate with shape (N, 3).
        pose (np.ndarray): 4x4 pose matrix representing rotation and translation.

    Returns:
        np.ndarray: 3D points in world coordinate with shape (N, 3).

    Raises:
        ValueError: If the input camera coordinate points or the pose matrix does not meet the required formats
        ValueError: If the output world coordinate points does not meet the required formats..

    Note:
        This function takes camera coordinate points and a pose matrix (4x4) representing the camera's position and orientation
        relative to the world coordinate system. It converts the points from camera coordinates to world coordinates using
        the provided pose.

        The rotation matrix (3x3) is extracted from the pose, representing the rotation from camera coordinates to world coordinates.
        The translation vector (3x1) is also extracted from the pose, representing the translation from camera coordinates to world coordinates.

        The function returns the world coordinate points (N, 3) after the conversion.
    """

    if not is_xyz_coordinate(cc_points):
        raise ValueError('The input camera coordinate points should be given in (N, 3)')

    if not is_pose_matrix(pose):
        raise ValueError('The input pose should be in correct format')

    # Extract the rotation and translation components from the pose matrix
    rot, trans = pose[:3, :3].reshape((3, 3)), pose[:3, -1].reshape((3, 1))

    # Convert camera coordinates to world coordinates using the inverse pose
    wc_points = (np.dot(rot, cc_points.T) + trans).T

    if not is_xyz_coordinate(wc_points):
        raise ValueError('The output world coordinate points should be given in (N, 3)')

    return wc_points


def camera_coordinate_to_uvd(
    points_3d: np.ndarray, intrinsic_matrix: np.ndarray, as_tuple: bool = True
) -> Union[np.ndarray, tuple]:
    """
    Convert 3D points from camera coordinates to UV coordinates and depth values using the camera intrinsic matrix.

    Args:
        points_3d (np.ndarray): Input 3D Points in camera coordinate with shape (N, 3).
        intrinsic_matrix (np.ndarray): Camera intrinsic matrix with shape (3, 3).

    Returns:
        tuple: UV coordinates (u, v) and depth values (d) with shapes (N,), (N,).

    Raises:
        ValueError: If the input camera coordinate points or the intrinsic matrix does not meet the required formats.
    """
    # define unit of depth value (1/m)
    DEPTH_UNIT = 1

    # Check if the input 3d points is in the correct format
    if not is_xyz_coordinate(points_3d):
        return None

    # Check if the input intrinsic matrix is in the correct format
    if not is_intrinsic_matrix(intrinsic_matrix):
        return None

    # [ud, vd, d](3 * N) = K(3 * 3) @ [Xc, Yc, Zc](3 * N)
    # Project the 3d points onto the image plane using the camera intrinsic
    # matrix
    projected_points = np.dot(intrinsic_matrix, points_3d.T)

    # Scale the projected points to obtain pixel coordinates (u, v) in the
    # image
    scaled_u = projected_points[0] / projected_points[2]
    scaled_v = projected_points[1] / projected_points[2]
    est_depth = projected_points[2]

    if as_tuple:
        return scaled_u, scaled_v, est_depth
    else:
        return np.column_stack((scaled_u, scaled_v, est_depth))


def uvd_to_camera_coordinate(uvd: tuple, intrinsic_matrix: np.ndarray) -> np.ndarray:
    """
    Convert UV coordinates and depth values to 3D points in camera coordinates using the camera intrinsic matrix.

    Args:
        uvd (tuple): Input UV coordinates (u, v) and depth values (d).
        intrinsic_matrix (np.ndarray): Camera intrinsic matrix with shape (3, 3).

    Returns:
        np.ndarray: Output 3D points in camera coordinate with shape (N, 3).

    Raises:
        ValueError: If the input UV coordinates or the intrinsic matrix does not meet the required formats.
    """
    uvd_array = np.asarray(uvd).reshape((-1, 3))

    # Remove points with depth value equal to 0 (invalid points) from the
    # uvd_array
    filtered_uvd_array = uvd_array[uvd_array[:, 2] != 0]

    # Scale the x and y coordinates by their respective depth values
    filtered_uvd_array[:, 0:2] *= filtered_uvd_array[:, [2]]

    # Solve linear equation K @ [Xc, Yc, Zc] = [ud, vd, d]
    points_3d = np.linalg.solve(intrinsic_matrix, filtered_uvd_array.T).T

    # Check if the input 3D points are in the correct format
    if not is_xyz_coordinate(points_3d):
        return None

    # Return the 3D points in the camera coordinate system
    return points_3d


def camara_coordinate_to_depth_image(
    points_3d: np.ndarray, intrinsic_matrix: np.ndarray, view_range: tuple
) -> np.ndarray:
    """
    Converts N * 3d points in camera coordinates to a depth map using the camera intrinsic matrix.

    Args:
        points_3d (numpy.ndarray): 3d xyz points data with shape (N, 3) representing 3D coordinates.
        intrinsic_matrix (numpy.ndarray): Camera intrinsic matrix with shape (3, 3).
        view_range (tuple): A tuple containing the FOV (Field of View).

    Returns:
        depth_image (numpy.ndarray): Depth map representing the depth values of the point cloud projected onto the image plane.
                      The shape of the depth map is determined by the view_range parameter.

    Raises:
        ValueError: If the input point cloud does not have the correct shape (N, 3).
        ValueError: If the input intrinsic matrix does not have the correct shape (3, 3).

    Note:
        This function first checks if the input point cloud and intrinsic matrix are in the correct format using
        the helper functions is_xyz_points and is_intrinsic_matrix. If any of the inputs is not in the correct format,
        the function returns None. Otherwise, it proceeds to project the 3D points onto the image plane using the
        camera intrinsic matrix. The depth map is then generated by associating each 3D point with its corresponding
        pixel coordinates (u, v) and depth value in the image. The resulting depth map contains the depth values of
        the 3D points projected onto the image plane, other invalid points will be removed.
    """

    scaled_u, scaled_v, depth = camera_coordinate_to_uvd(points_3d, intrinsic_matrix)

    # Round the scaled uv coordinates and convert to integer
    u = np.round(scaled_u).astype(int)
    v = np.round(scaled_v).astype(int)

    # rows and columns in the FOV (Field of View)
    row, col = view_range

    # Find the valid indices where (u, v) are within the image range
    valid_idx = (u >= 0) & (v >= 0) & (u < col) & (v < row)

    # Retrieve the valid pixel coordinates (u, v) and their corresponding
    # depth values from the projected points
    u_valid = u[valid_idx]
    v_valid = v[valid_idx]
    depth_valid = depth[valid_idx]

    # sort the depth values from large to small
    depth_sort_idx = np.argsort(depth_valid)[::-1]

    # Initialize the depth map with zeros
    depth_image = np.zeros(view_range, dtype=np.float32)

    # Populate the depth map with the depth values of the valid projected points
    # point with high depth-values in the same (u, v) coordinate will be
    # covered by low depth values
    depth_image[v_valid[depth_sort_idx], u_valid[depth_sort_idx]] = depth_valid[
        depth_sort_idx
    ]

    return depth_image


def depth_image_to_camera_coordinate(
    depth_image: np.ndarray, intrinsic_matrix: np.ndarray
) -> np.ndarray:
    """
    Convert a depth image to 3D points in the camera coordinate system.

    Args:
        depth_image (np.ndarray): A 2D array representing the depth information.
        intrinsic_matrix (np.ndarray): The camera's intrinsic matrix used to convert pixel coordinates to camera coordinates.

    Returns:
        np.ndarray: An array containing 3D points in the camera coordinate system.
    """

    # Check if the input intrinsic matrix is in the correct format
    if not is_intrinsic_matrix(intrinsic_matrix):
        return None

    # Get the range of the depth image
    rows, cols = depth_image.shape
    v = np.arange(rows)
    u = np.arange(cols)

    # Create a meshgrid of pixel coordinates (uu and vv)
    uu, vv = np.meshgrid(u, v)

    # Stack pixel coordinates and depth values into a 3D array (uvd_image)
    uvd_image = np.dstack((uu, vv, depth_image))

    # Reshape the 3D array into a 2D array (uvd_array) for further processing
    uvd_array = uvd_image.reshape((-1, 3))

    # Remove points with depth value equal to 0 (invalid points) from the
    # uvd_array
    filtered_uvd_array = uvd_array[uvd_array[:, 2] != 0]

    # Scale the x and y coordinates by their respective depth values
    filtered_uvd_array[:, 0:2] *= filtered_uvd_array[:, [2]]

    # Solve linear equation K @ [Xc, Yc, Zc] = [ud, vd, d]
    points_3d = np.linalg.solve(intrinsic_matrix, filtered_uvd_array.T).T

    # Check if the input 3D points are in the correct format
    if not is_xyz_coordinate(points_3d):
        return None

    # Return the 3D points in the camera coordinate system
    return points_3d


def depth_image_to_world_coordinate_image(
    depth_image: np.ndarray,
    intrinsic_matrix: np.ndarray,
    pose: np.ndarray,
    fill: Any = 0,
) -> np.ndarray:
    """
    Convert a depth image to a world coordinate image.

    Parameters:
        depth_image (np.ndarray): Input depth image (HxW).
        intrinsic_matrix (np.ndarray): 3x3 camera intrinsic matrix.
        pose (np.ndarray): 4x4 camera pose matrix (world-to-camera transformation).
        fill (Any, optional): Value to fill invalid depth pixels with (default is 0).

    Returns:
        np.ndarray: World coordinate image (HxWx3), where each pixel contains the (X, Y, Z) world coordinates.

    Raises:
        ValueError: If the intrinsic_matrix and the pose matrix does not meet the required formats.
    """

    # Check if the input intrinsic matrix is in the correct format
    if not is_intrinsic_matrix(intrinsic_matrix):
        return None

    if not is_pose_matrix(pose):
        raise ValueError('The input pose should be in correct format')

    # Get the range of the depth image
    rows, cols = depth_image.shape
    v = np.arange(rows)
    u = np.arange(cols)

    # Create a meshgrid of pixel coordinates (uu and vv)
    uu, vv = np.meshgrid(u, v)

    # Stack pixel coordinates and depth values into a 3D array (uvd_image)
    uvd_image = np.dstack((uu, vv, depth_image))

    # Reshape the 3D array into a 2D array (uvd_array) for further processing
    uvd_array = uvd_image.reshape((-1, 3))

    # Scale the x and y coordinates by their respective depth values
    uvd_array[:, 0:2] *= uvd_array[:, [2]]

    # Solve linear equation K @ [Xc, Yc, Zc] = [ud, vd, d]
    points_3d_cc = np.linalg.solve(intrinsic_matrix, uvd_array.T).T

    # Convert camera coordinate to world coordinate
    points_3d_wc = camera_coordinate_to_world_coordinate(points_3d_cc, pose)

    # Reshape the result back to a world coordinate image
    wc_image = points_3d_wc.reshape((rows, cols, 3))

    # Fill invalid depth pixels with the specified fill value
    wc_image[np.where(depth_image == 0)] = fill

    return wc_image


def world_coordinate_image_to_depth_image(
    wc_image: np.ndarray, intrinsic_matrix: np.ndarray, pose: np.ndarray
) -> np.ndarray:
    """
    Convert a world coordinate image to a depth image.

    Parameters:
        wc_image (np.ndarray): Input world coordinate image (HxWx3).
        intrinsic (np.ndarray): 3x3 camera intrinsic matrix.
        pose (np.ndarray): 4x4 camera pose matrix (world-to-camera transformation).

    Returns:
        np.ndarray: Depth image (HxW), where each pixel contains the depth value.

    Raises:
        ValueError: If the intrinsic_matrix and the pose matrix does not meet the required formats.
    """

    # Check if the input intrinsic matrix is in the correct format
    if not is_intrinsic_matrix(intrinsic_matrix):
        return None

    if not is_pose_matrix(pose):
        raise ValueError('The input pose should be in correct format')

    h, w, _ = wc_image.shape

    # Get the indices of invalid world coordinate pixels (where [0, 0, 0] is
    # present)
    invalid_indices = np.where((wc_image == [0, 0, 0]).all(axis=-1))

    # Reshape the world coordinate image to a 2D array for processing
    points_3d_wc = wc_image.reshape((-1, 3))

    # Convert world coordinate to camera coordinate
    points_3d_cc = world_coordinate_to_camera_coordinate(points_3d_wc, pose)

    # Convert camera coordinate to depth image
    depth_image = camara_coordinate_to_depth_image(
        points_3d_cc, intrinsic_matrix, (h, w)
    )

    # Set the depth values of invalid pixels to 0
    depth_image[invalid_indices] = 0

    return depth_image


def world_coordinate_to_uvd(
    wc_points: np.ndarray, intrinsic_matrix: np.ndarray, pose: np.ndarray
) -> tuple:
    """
    Convert world coordinates to UV coordinates and depth values using camera intrinsic matrix and pose.

    Args:
        wc_points (np.ndarray): Input 3D points in world coordinate with shape (N, 3).
        intrinsic_matrix (np.ndarray): Camera intrinsic matrix with shape (3, 3).
        pose (np.ndarray): 4x4 pose matrix representing rotation and translation.

    Returns:
        tuple: UV coordinates (u, v) and depth values (d) with shapes (N,), (N,).

    Raises:
        ValueError: If the input world coordinate points or the pose matrix does not meet the required formats.
    """
    cc_points = world_coordinate_to_camera_coordinate(wc_points, pose)
    return camera_coordinate_to_uvd(cc_points, intrinsic_matrix)


def uvd_to_world_coordinate(
    uvd: tuple, intrinsic_matrix: np.ndarray, pose: np.ndarray
) -> np.ndarray:
    """
    Convert UV coordinates and depth values to world coordinates using camera intrinsic matrix and pose.

    Args:
        uvd (tuple): Input UV coordinates (u, v) and depth values (d).
        intrinsic_matrix (np.ndarray): Camera intrinsic matrix with shape (3, 3).
        pose (np.ndarray): 4x4 pose matrix representing rotation and translation.

    Returns:
        np.ndarray: Output 3D points in world coordinate with shape (N, 3).

    Raises:
        ValueError: If the input UV coordinates or the intrinsic matrix does not meet the required formats.
    """
    cc_points = uvd_to_camera_coordinate(uvd, intrinsic_matrix)
    return camera_coordinate_to_world_coordinate(cc_points, pose)
