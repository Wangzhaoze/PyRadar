import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from radar_gs_utils import *
from radar_gs_datamodule import ColoradarTI1843BoostConfig, ColoradarTI2243Config
from scipy.fft import fft, fftshift

# ==== 1. 场景点云（球体） ====
def generate_sphere_point_cloud(radius=1.0, num_points=10000, center=np.array([0, 0, 0])):
    phi = np.random.uniform(0, np.pi, num_points)
    theta = np.random.uniform(0, 2 * np.pi, num_points)

    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)

    points = np.stack([x, y, z], axis=1) + center
    return points


# ==== 2. 生成雷达轨迹 ====
def generate_linear_trajectory(n_frames=60):
    poses = []
    for delta_trans in np.linspace(0, 0.1, n_frames):

        T = np.eye(4)
        T[:3, 3] = [delta_trans, 0, 0]  # Move along x-axis
        poses.append(T)
    return np.stack(poses, axis=0)



# ==== 3. Range image 函数 ====

def compute_range_image_with_visibility(
    points,                 # [N, 3] world coords
    pose,                   # [4, 4] antenna pose (world to antenna)
    az_fov=90, el_fov=90,   # Field of view in degrees (+/-)
    az_res=1, el_res=1,     # Resolution in degrees
    point_radius=0          # Optional radius in degrees
):
    # Step 1: Transform points to antenna frame
    N = points.shape[0]
    points_h = np.hstack((points, np.ones((N, 1))))  # [N, 4]
    points_local = (np.linalg.inv(pose) @ points_h.T).T[:, :3]  # [N, 3]

    x, y, z = points_local[:, 0], points_local[:, 1], points_local[:, 2]
    r = np.linalg.norm(points_local, axis=1)

    # Filter: keep only points in front of antenna (positive z)
    valid = x < 1000
    x, y, z, r = x[valid], y[valid], z[valid], r[valid]
    indices = np.nonzero(valid)[0]  # Keep track of which original points are valid

    # Step 2: Compute azimuth and elevation in degrees
    az = np.degrees(np.arctan2(y, x))      # [-180, 180]
    el = np.degrees(np.arcsin(z / r))      # [-90, 90]

    # Step 3: Map angles to image coordinates
    az_min, az_max = -az_fov, az_fov
    el_min, el_max = -el_fov, el_fov

    az_bins = int((az_max - az_min) / az_res)
    el_bins = int((el_max - el_min) / el_res)

    range_image = np.full((el_bins, az_bins), np.inf)
    index_image = np.full((el_bins, az_bins), -1)  # Track point indices for visibility
    visibility_mask = np.zeros(N, dtype=bool)

    # Step 4: Rasterization with optional radius (smooth)
    for i, (a, e, dist) in enumerate(zip(az, el, r)):
        if az_min <= a <= az_max and el_min <= e <= el_max:
            az_idx = int((a - az_min) / az_res)
            el_idx = int((e - el_min) / el_res)

            radius_bin = int(point_radius / az_res)

            # apply smoothing if radius > 0
            for da in range(-radius_bin, radius_bin + 1):
                for de in range(-radius_bin, radius_bin + 1):
                    azi = az_idx + da
                    eli = el_idx + de
                    if 0 <= azi < az_bins and 0 <= eli < el_bins:
                        if dist < range_image[eli, azi]:
                            range_image[eli, azi] = dist
                            index_image[eli, azi] = indices[i]

    # Step 5: mark visible points
    unique_visible_indices = np.unique(index_image[index_image >= 0])
    visibility_mask[unique_visible_indices] = True

    return range_image, visibility_mask



def radar_module(scene_pcd, RX_pose, rcs_spectrum, radar_config: ColoradarTI1843BoostConfig):
    range_image, visibility_mask = compute_range_image_with_visibility(
    scene_pcd, RX_pose,
    az_fov=90, el_fov=90,
    az_res=1, el_res=1,
    point_radius=1.0
    )

    range = range_image.reshape((-1, 1))
    out_of_range_mask = (range > 28)  # [32400]
    range[out_of_range_mask] = 0  # Set out of range values to 0

    rcs_real = rcs_spectrum[0].reshape((-1, 1))  # [32400, 1]
    rcs_imag = rcs_spectrum[1].reshape((-1, 1))  # [32400, 1]

    rcs_real[out_of_range_mask] = 0
    rcs_imag[out_of_range_mask] = 0

    f_R = 2 * radar_config.chirpSlope * range / 3e8
    f_D = 0
    Phi_0 = 2 * range / radar_config.waveLength

    sampleTimeArray = radar_config.sampleTimeArray.reshape((1, -1))  # [1, numSamples]

    S_IF = (rcs_real + 1j * rcs_imag) * np.exp(1j * 2 * np.pi * (((f_R + f_D) * sampleTimeArray + Phi_0)))
    S_IF = np.sum(S_IF, axis=0, keepdims=False)  # [1, numSamples, 2]

    # 4 * torch.pi * chirpSlope * pred_depth.view(-1, 1) * sampleTimeArray  / 3e8 + 2 * pred_depth.view(-1, 1) / waveLength
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



radar_config = ColoradarTI1843BoostConfig()
sim_adc = np.zeros((radar_config.numChirpsPerFrame, radar_config.numADCSamples), dtype=np.complex64)  # Simulated ADC output
scene_points = generate_sphere_point_cloud(center=np.array([10, 0, 0]), radius=1.0)
trajectory = generate_linear_trajectory(n_frames=radar_config.numChirpsPerFrame)
dummy_rcs_spectrum = np.zeros((2, 180, 180))  # Dummy RCS spectrum
dummy_rcs_spectrum[0] = 1.0  # Set all RCS values to 1.0

for idxChirp in range(radar_config.numChirpsPerFrame):
    RX_pose = trajectory[idxChirp]  # Get the antenna pose for this chirp
    sim_adc[idxChirp] = radar_module(
        scene_points, 
        RX_pose,
        rcs_spectrum=dummy_rcs_spectrum,  # Dummy RCS spectrum
        radar_config=radar_config
    )

sim_spectrum = range_doppler_map(adc_cube=sim_adc[..., np.newaxis], IdxChirps=0, IdxSamples=1)  # Take magnitude of the complex ADC output
# ==== 4. 可视化 Range Image ===
plt.imshow(np.abs(sim_spectrum), cmap='jet')
plt.show()
print("Simulated ADC output shape:", sim_adc.shape)

# plt.imshow(range_image, cmap='jet', vmin=0, vmax=10, origin='lower')
# plt.colorbar(label='Range (m)')
# plt.title("Range Image at Antenna Pose")
# plt.xlabel("Azimuth (deg)")
# plt.ylabel("Elevation (deg)")
# plt.show()


# from mpl_toolkits.mplot3d import Axes3D

# def draw_fov_cone(ax, pose, fov_deg=45, length=3.0, color='cyan', resolution=30):
#     # Sample directions on the cone surface (angle from center axis)
#     theta = np.linspace(0, 2 * np.pi, resolution)
#     half_angle = np.radians(fov_deg)

#     # Circle on base of cone in local coordinates
#     r = np.tan(half_angle) * length
#     x = np.full_like(theta, length)  # forward direction is local Z+
#     y = r * np.cos(theta)
#     z = r * np.sin(theta)

#     # Stack into shape (N, 3), directions in local coordinates
#     local_pts = np.stack([x, y, z], axis=1)

#     # Add the origin (apex of cone)
#     local_pts = np.vstack(([[0, 0, 0]], local_pts))  # shape (N+1, 3)

#     # Transform to world coordinates
#     local_pts_h = np.hstack([local_pts, np.ones((local_pts.shape[0], 1))])  # [N+1, 4]
#     world_pts = (pose @ local_pts_h.T).T[:, :3]

#     # Triangulate cone surface
#     from mpl_toolkits.mplot3d.art3d import Poly3DCollection
#     apex = world_pts[0]
#     base_pts = world_pts[1:]

#     verts = [[apex, base_pts[i], base_pts[(i+1)%len(base_pts)]] for i in range(len(base_pts))]
#     cone = Poly3DCollection(verts, color=color, alpha=0.2)
#     ax.add_collection3d(cone)


# def draw_coordinate_frame(ax, pose, scale=0.5):
#     origin = pose[:3, 3]
#     x_axis = pose[:3, 0] * scale + origin
#     y_axis = pose[:3, 1] * scale + origin
#     z_axis = pose[:3, 2] * scale + origin

#     ax.plot([origin[0], x_axis[0]], [origin[1], x_axis[1]], [origin[2], x_axis[2]], 'r')
#     ax.plot([origin[0], y_axis[0]], [origin[1], y_axis[1]], [origin[2], y_axis[2]], 'g')
#     ax.plot([origin[0], z_axis[0]], [origin[1], z_axis[1]], [origin[2], z_axis[2]], 'b')



# from matplotlib import cm
# def animate_range_images_with_3d(points, trajectory, fov=45, resolution=1, point_radius=1):
#     fig = plt.figure(figsize=(12, 6))
#     ax_img = fig.add_subplot(1, 2, 1)
#     ax_img.set_title("Range Image")
#     extent = [-fov, fov, -fov, fov]
#     im = ax_img.imshow(np.zeros((2*fov, 2*fov)), cmap='jet', vmin=0, vmax=10, extent=extent, origin='lower')
#     ax_img.set_xlabel("Azimuth (deg)")
#     ax_img.set_ylabel("Elevation (deg)")

#     ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
#     ax_3d.set_title("3D Scene")
#     ax_3d.set_xlim([0, 12])
#     ax_3d.set_ylim([-3, 3])
#     ax_3d.set_zlim([-3, 3])
#     ax_3d.view_init(elev=30, azim=120)

#     def update(i):
#         ax_3d.cla()
#         ax_3d.set_title(f"3D Scene - Frame {i+1}")
#         ax_3d.set_xlim([0, 12])
#         ax_3d.set_ylim([-3, 3])
#         ax_3d.set_zlim([-3, 3])
#         ax_3d.view_init(elev=30, azim=150)

#         ax_3d.scatter(points[:, 0], points[:, 1], points[:, 2], c='gray', s=1, alpha=0.3)

#         pose = trajectory[i]
#         draw_coordinate_frame(ax_3d, pose)
#         draw_fov_cone(ax_3d, pose, fov_deg=fov, length=3.0)

#         range_img, _ = compute_range_image_with_visibility(
#             points, pose,
#             az_fov=fov, el_fov=fov,
#             az_res=resolution, el_res=resolution,
#             point_radius=point_radius
#         )
#         img = np.where(np.isfinite(range_img), range_img, 0)
#         im.set_data(img)
#         ax_img.set_title(f"Range Image - Frame {i+1}")
#         return [im]

#     ani = animation.FuncAnimation(fig, update, frames=len(trajectory), interval=100, blit=False)
#     plt.tight_layout()
#     plt.show()


# # ==== 4. 运行实验 ====
# animate_range_images_with_3d(scene_points, trajectory)
