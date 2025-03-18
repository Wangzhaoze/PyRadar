"""Visualization Tools of Radar Signal."""

import matplotlib.pyplot as plt
from typing import Optional, Union
import open3d as o3d

import numpy as np


def show_2D_heat_map(
    spectrum: np.ndarray,
    figsize: tuple = (8, 6),
    xaxis: Union[np.ndarray, list] = None,
    yaxis: Union[np.ndarray, list] = None,
    cmap: str = 'jet',
    xlabel: str = 'X-axis',
    ylabel: str = 'Y-axis',
    title: str = 'Heat Map',
    colorbar_label: str = 'Magnitude',
):
    """
    Display a heat map of a given spectrum.

    Parameters:
    - spectrum (np.ndarray): 2D numpy array of complex values representing the spectrum.
    - xaxis (np.ndarray, optional): Values for the x-axis. Defaults to index-based values.
    - yaxis (np.ndarray, optional): Values for the y-axis. Defaults to index-based values.
    - cmap (str, optional): Colormap for the heat map. Default is 'jet'.
    - xlabel (str, optional): Label for the x-axis. Default is 'X-axis'.
    - ylabel (str, optional): Label for the y-axis. Default is 'Y-axis'.
    - title (str, optional): Title of the plot. Default is 'Heat Map'.
    - colorbar_label (str, optional): Label for the colorbar. Default is 'Magnitude'.

    Raises:
    - ValueError: If input spectrum is not a 2D numpy array of complex values.
    """
    # Validate inputs
    if not isinstance(spectrum, np.ndarray):
        raise ValueError('Input spectrum must be a numpy array.')
    # if np.issubdtype(spectrum.dtype, np.complexfloating) or isinstance(spectrum, complex):
    #     raise ValueError(f'Input spectrum must contain complex numbers, but it is {spectrum.dtype}.')
    if spectrum.ndim != 2:
        raise ValueError('Input spectrum must be a 2D array.')

    # Set x and y axis extents
    xaxis = xaxis if xaxis is not None else np.arange(spectrum.shape[1])
    yaxis = yaxis if yaxis is not None else np.arange(spectrum.shape[0])

    # Plot the heat map
    plt.figure(figsize=figsize)
    plt.imshow(
        np.abs(spectrum),
        # 20.0 * np.log10( np.abs(spectrum) ),
        cmap=cmap,
        extent=[xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]],
        aspect='auto',
    )
    plt.colorbar(label=colorbar_label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def open3d_pointcloud_to_numpy(pcd):
    """
    Convert Open3D point cloud to numpy array.

    Args:
        pcd (open3d.geometry.PointCloud): The Open3D point cloud.

    Returns:
        np.ndarray: The converted numpy array.
    """
    if isinstance(pcd, o3d.geometry.PointCloud):
        pcd_array = np.asarray(pcd.points)
    else:
        raise ValueError('Input object should be an open3d.geometry.PointCloud object')

    return pcd_array


def numpy_to_open3d_pointcloud(np_points):
    """
    Convert numpy array to Open3D point cloud.

    Args:
        np_points (np.ndarray): The numpy array of points.

    Returns:
        open3d.geometry.PointCloud: The created Open3D point cloud.
    """
    if isinstance(np_points, np.ndarray):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np_points)
    else:
        raise ValueError('Input object should be a numpy.ndarray object')

    if colors is not None and len(colors) != 0:
        if np.max(colors) > 1:
            colors = (colors / np.max(colors)).astype(np.float32)
        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def visualize_point_cloud(
    pcd: Union[o3d.geometry.PointCloud, list], title='point cloud'
):
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


def visualize_ray_tracing(point_cloud: Union[np.ndarray, list]) -> None:
    """
    Visualize the paths that connect the origin point to each point in the point cloud.

    Args:
        point_cloud (Union[np.ndarray, List[List[float]]]): A list or array of n*3 point cloud.
    """
    # Convert the point cloud to a numpy array if it is a list
    if isinstance(point_cloud, list):
        point_cloud = np.array(point_cloud)

    # Ensure the point cloud is an n*3 array
    assert point_cloud.shape[1] == 3, 'Point cloud must be of shape n*3'

    # Convert the point cloud to an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    # Create lines that connect the origin to each point
    origin = np.array([[0, 0, 0]])
    points = np.vstack((origin, point_cloud))
    lines = [[0, i + 1] for i in range(point_cloud.shape[0])]

    # Create a LineSet object
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)

    # Optionally, set colors for the lines
    colors = [[1, 0, 0] for _ in range(len(lines))]  # Red color for all lines
    line_set.colors = o3d.utility.Vector3dVector(colors)

    # Visualize the point cloud and the lines
    o3d.visualization.draw_geometries([pcd, line_set, draw_xyz_frame()])


def draw_xyz_frame():
    """
    Create an XYZ frame.

    Returns:
        open3d.geometry.TriangleMesh: The created XYZ frame.
    """
    return o3d.geometry.TriangleMesh.create_coordinate_frame()
