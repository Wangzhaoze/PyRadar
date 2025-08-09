import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from scipy.fft import fft, fftshift
from typing import Optional

# ==== 1. 场景点云（球体） ====
def generate_sphere_point_cloud(radius=1.0, num_points=10000, center=np.array([0, 0, 0])):
    phi = np.random.uniform(0, np.pi, num_points)
    theta = np.random.uniform(0, 2 * np.pi, num_points)

    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)

    points = np.stack([x, y, z], axis=1) + center

    # generate RCS for each point
    rcs = np.ones((points.shape[0], 1)) + 1j * np.zeros((points.shape[0], 1))  # Dummy RCS values, can be replaced with actual RCS data
    return points, rcs


# ==== 2. 生成雷达轨迹 ====
def generate_linear_trajectory(n_frames=60):
    poses = []
    for delta_trans in np.linspace(0, 0.05, n_frames):

        T = np.eye(4)
        T[:3, 3] = [delta_trans, 0, 0]  # Move along x-axis
        poses.append(T)
    return np.stack(poses, axis=0)


def xyz2aer(points: np.ndarray, as_degrees: bool = True) -> np.ndarray:
    """
    Convert 3D points from Cartesian coordinates (x, y, z) to spherical coordinates (azimuth, elevation, range).

    Args:
        points (np.ndarray): Nx3 array of points in Cartesian coordinates.

    Returns:
        np.ndarray: Nx3 array of points in spherical coordinates (azimuth, elevation, range).
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    range = np.sqrt(x**2 + y**2 + z**2)
    azimuth = np.arctan2(-y, x)
    elevation = np.arcsin(z / range)

    if as_degrees:
        azimuth = np.degrees(azimuth)
        elevation = np.degrees(elevation)
    return np.column_stack((azimuth, elevation, range))

def compute_range_image_with_visibility(
    xyz_abs: np.ndarray,
    antenna_pose: np.ndarray, # [4, 4] antenna pose (world to antenna)
    features: Optional[np.ndarray] = None,  # [N, 3] world coords                  
    az_fov: float=180, 
    el_fov: float=180,
    max_range: float = 50,   # Field of view in degrees
    az_res: float = 1, 
    el_res: float = 1,     # Resolution in degrees
    point_radius: float = 0          # Optional radius in degrees
):
    points_h = np.hstack((xyz_abs, np.ones((xyz_abs.shape[0], 1))))
    points_local = (np.linalg.inv(antenna_pose) @ points_h.T).T[:, :3]

    visibility_mask = np.ones(points_local.shape[0], dtype=bool)
    azimuth, elevation, range = xyz2aer(points_local, as_degrees=True).T

    # filter points based on azimuth and elevation FOV
    azimuth_mask = (azimuth >= -az_fov) & (azimuth <= az_fov)
    elevation_mask = (elevation >= -el_fov) & (elevation <= el_fov)
    range_mask = (range > 0) & (range <= max_range)
    visibility_mask &= azimuth_mask & elevation_mask & range_mask 

    azimuth = azimuth[visibility_mask]
    elevation = elevation[visibility_mask]
    range = range[visibility_mask]
    if features is not None:
        features = features[visibility_mask]

    # sorted azimuth, elevation, and range from range max to min
    sorted_indices = np.argsort(range)[::-1]
    azimuth = azimuth[sorted_indices]
    elevation = elevation[sorted_indices]
    range = range[sorted_indices]

    # Convert azimuth and elevation to pixel coordinates
    azimuth_pixel = np.floor((azimuth + az_fov) / az_res).astype(int)
    elevation_pixel = np.floor((elevation + el_fov) / el_res).astype(int)
    range_image = np.full(
        (int(az_fov * 2 / az_res), int(el_fov * 2 / el_res)), 
        np.inf, 
        dtype=np.float32
    )

    if features is not None:
        features = features[visibility_mask]
        features = features[sorted_indices]
        feature_image = np.full(
            (int(az_fov * 2 / az_res), int(el_fov * 2 / el_res), features.shape[1]),
            0.0,
            dtype=features.dtype
        )
        feature_image[elevation_pixel, azimuth_pixel] = features
    else:
        feature_image = None

    range_image[elevation_pixel, azimuth_pixel] = range

    return range_image, visibility_mask, feature_image


def radar_simulation(
        scene_pcd, 
        scene_pcd_rcs,
        RX_pose, 
        radar_config
        ):
    
    range_image, visibility_mask, rcs_spectrum = compute_range_image_with_visibility(
        scene_pcd, 
        RX_pose,
        features=scene_pcd_rcs,
        az_fov=90, 
        el_fov=90,
        az_res=1, 
        el_res=1,
        point_radius=1.0
    )

    range = range_image.reshape((-1, 1))
    out_of_range_mask = (range > 28)  # [32400]
    range[out_of_range_mask] = 0  # Set out of range values to 0

    rcs_real = np.real(rcs_spectrum).reshape((-1, 1))  # [32400, 1]
    rcs_imag = np.imag(rcs_spectrum).reshape((-1, 1))  # [32400, 1]

    rcs_real[out_of_range_mask] = 0
    rcs_imag[out_of_range_mask] = 0

    f_R = 2 * radar_config.chirpSlope * range / 3e8
    f_D = 0
    Phi_0 = 2 * range / radar_config.waveLength

    sampleTimeArray = radar_config.sampleTimeArray.reshape((1, -1))  # [1, numSamples]

    S_IF = (rcs_real + 1j * rcs_imag) * np.exp(1j * 2 * np.pi * (((f_R + f_D) * sampleTimeArray + Phi_0)))
    S_IF = np.sum(S_IF, axis=0, keepdims=False)  # [1, numSamples, 2]

    # S_IF = np.sum(np.real(S_IF), axis=0, keepdims=False) + 1j * np.sum(np.imag(S_IF), axis=0, keepdims=False)  # [numSamples, 2]
    # S_IF = 4 * torch.pi * chirpSlope * pred_depth.view(-1, 1) * sampleTimeArray  / 3e8 + 2 * pred_depth.view(-1, 1) / waveLength
    return S_IF


def range_doppler_fft(
    adc_cube: np.ndarray,
    IdxSamples: int = 1,
    IdxChirps: int = 2,
    num_workers: Optional[int] = None,
) -> np.ndarray:
    """
    Perform Range FFT followed by Doppler FFT to generate a Range-Doppler map.

    Args:
        adc_cube (np.ndarray): (numSamples, numChirps, numAntennas) shape 3D array with complex ADC (Analog-to-Digital Converter) data.
        IdxSamples (int): Axis index for the range dimension (default: 1).
        IdxChirps (int): Axis index for the Doppler dimension (default: 2).

    Returns:
        np.ndarray: The Range-Doppler map with both range and Doppler dimensions transformed to the frequency domain.
    """
    # Ensure input is a 3D numpy array
    if not isinstance(adc_cube, np.ndarray):
        raise ValueError('Input must be a numpy array.')
    if adc_cube.ndim != 3:
        raise ValueError('Input array must have exactly three dimensions.')

    # Perform Range FFT along the range axis
    range_spectrum = fft(adc_cube, axis=IdxSamples, workers=num_workers)

    # Perform Doppler FFT along the Doppler axis and shift the zero frequency
    # component to the center
    range_doppler_spectrum = fftshift(
        fft(range_spectrum, axis=IdxChirps, workers=num_workers), axes=IdxChirps
    )
    return range_doppler_spectrum

def range_doppler_map(adc_cube: np.ndarray,
                      IdxSamples: int = 0,
                      IdxChirps: int = 1,
                      IdxVirtualAntennas: int = 2):
    """
    Computes the Range-Doppler Map from the given ADC data cube by performing a Range-Doppler FFT.

    The function calculates the Range-Doppler Map from the 3D ADC data cube, which typically represents
    samples, chirps, and virtual antennas. The map shows the range on the horizontal axis (0 to Rmax)
    and velocity on the vertical axis (from -Vmin to Vmax).

    Parameters:
    - adc_cube (np.ndarray): 3D numpy array containing the ADC data cube. The dimensions represent samples, chirps, and virtual antennas.
    - IdxSamples (int): Index of the samples dimension in the `adc_cube`. Default is 0.
    - IdxChirps (int): Index of the chirps dimension in the `adc_cube`. Default is 1.
    - IdxVirtualAntennas (int): Index of the virtual antennas dimension in the `adc_cube`. Default is 2.

    Returns:
    - range_doppler_map (np.ndarray): 2D Range-Doppler map where:
      - The x-axis represents range from 0 to Rmax.
      - The y-axis represents velocity from -Vmin to Vmax.

      Visual representation:

                        #################################################
                        #               Range-Doppler Map               #
                        #################################################
                        #                                               #
                        #   Velocity                                    #
                        #                                               #
                        #   +Vmax    ▲                                  #
                        #            |                                  #
                        #            |                                  #
                        #            |                                  #
                        #   0 Hz     |                                  #
                        #            |                                  #
                        #            |                                  #
                        #            |                                  #
                        #   -Vmin     ————————————————————————▷         #
                        #                                               #
                        #           0m         Range           Rmax     #
                        #################################################
    """
    # Reorder the input `adc_cube` dimensions so chirps, samples, and virtual antennas are in (0, 1, 2) order.
    adc_cube = np.transpose(adc_cube, (IdxChirps, IdxSamples, IdxVirtualAntennas))

    # Perform Range-Doppler FFT on the ADC cube. Compute the spectrum along samples and chirps dimensions.
    range_doppler_spectrum = range_doppler_fft(adc_cube, IdxSamples=1, IdxChirps=0)

    # Extract the first virtual antenna slice (index 0) along the virtual antenna axis.
    range_doppler_map = range_doppler_spectrum.take(indices=0, axis=2)

    # Flip the velocity axis (axis 0) so the maximum velocity appears at the top.
    range_doppler_map = np.flip(range_doppler_map, axis=0)

    return range_doppler_map




from mpl_toolkits.mplot3d import Axes3D

def draw_fov_cone(ax, pose, fov_deg=45, length=3.0, color='cyan', resolution=30):
    # Sample directions on the cone surface (angle from center axis)
    theta = np.linspace(0, 2 * np.pi, resolution)
    half_angle = np.radians(fov_deg)

    # Circle on base of cone in local coordinates
    r = np.tan(half_angle) * length
    x = np.full_like(theta, length)  # forward direction is local Z+
    y = r * np.cos(theta)
    z = r * np.sin(theta)

    # Stack into shape (N, 3), directions in local coordinates
    local_pts = np.stack([x, y, z], axis=1)

    # Add the origin (apex of cone)
    local_pts = np.vstack(([[0, 0, 0]], local_pts))  # shape (N+1, 3)

    # Transform to world coordinates
    local_pts_h = np.hstack([local_pts, np.ones((local_pts.shape[0], 1))])  # [N+1, 4]
    world_pts = (pose @ local_pts_h.T).T[:, :3]

    # Triangulate cone surface
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    apex = world_pts[0]
    base_pts = world_pts[1:]

    verts = [[apex, base_pts[i], base_pts[(i+1)%len(base_pts)]] for i in range(len(base_pts))]
    cone = Poly3DCollection(verts, color=color, alpha=0.2)
    ax.add_collection3d(cone)


def draw_coordinate_frame(ax, pose, scale=0.5):
    origin = pose[:3, 3]
    x_axis = pose[:3, 0] * scale + origin
    y_axis = pose[:3, 1] * scale + origin
    z_axis = pose[:3, 2] * scale + origin

    ax.plot([origin[0], x_axis[0]], [origin[1], x_axis[1]], [origin[2], x_axis[2]], 'r')
    ax.plot([origin[0], y_axis[0]], [origin[1], y_axis[1]], [origin[2], y_axis[2]], 'g')
    ax.plot([origin[0], z_axis[0]], [origin[1], z_axis[1]], [origin[2], z_axis[2]], 'b')



from matplotlib import cm
def animate_range_images_with_3d(points, trajectory, fov=45, resolution=1, point_radius=1):
    fig = plt.figure(figsize=(12, 6))
    ax_img = fig.add_subplot(1, 2, 1)
    ax_img.set_title("Range Image")
    extent = [-fov, fov, -fov, fov]
    im = ax_img.imshow(np.zeros((2*fov, 2*fov)), cmap='jet', vmin=0, vmax=10, extent=extent, origin='lower')
    ax_img.set_xlabel("Azimuth (deg)")
    ax_img.set_ylabel("Elevation (deg)")

    ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
    ax_3d.set_title("3D Scene")
    ax_3d.set_xlim([0, 12])
    ax_3d.set_ylim([-3, 3])
    ax_3d.set_zlim([-3, 3])
    ax_3d.view_init(elev=30, azim=120)

    def update(i):
        ax_3d.cla()
        ax_3d.set_title(f"3D Scene - Frame {i+1}")
        ax_3d.set_xlim([0, 12])
        ax_3d.set_ylim([-3, 3])
        ax_3d.set_zlim([-3, 3])
        ax_3d.view_init(elev=30, azim=150)

        ax_3d.scatter(points[:, 0], points[:, 1], points[:, 2], c='gray', s=1, alpha=0.3)

        pose = trajectory[i]
        draw_coordinate_frame(ax_3d, pose)
        # draw_fov_cone(ax_3d, pose, fov_deg=fov, length=3.0)

        range_image, visibility_mask, rcs_spectrum = compute_range_image_with_visibility(
            points, 
            pose,
            features=None,
            az_fov=90, 
            el_fov=90,
            az_res=1, 
            el_res=1,
            point_radius=1.0
        )
        img = np.where(np.isfinite(range_image), range_image, 0)
        im.set_data(img)
        ax_img.set_title(f"Range Image - Frame {i+1}")
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=len(trajectory), interval=100, blit=False)
    plt.tight_layout()
    plt.show()






# plt.imshow(range_image, cmap='jet', vmin=0, vmax=10, origin='lower')
# plt.colorbar(label='Range (m)')
# plt.title("Range Image at Antenna Pose")
# plt.xlabel("Azimuth (deg)")
# plt.ylabel("Elevation (deg)")
# plt.show()


# ==== 4. 运行实验 ====

