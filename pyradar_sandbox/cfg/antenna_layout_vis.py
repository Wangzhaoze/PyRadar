import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def create_ula(num_elements, spacing, axis='x'):
    positions = np.zeros((num_elements, 3))
    coords = np.linspace(0, (num_elements - 1) * spacing, num_elements)
    if axis == 'x':
        positions[:, 0] = coords
    elif axis == 'y':
        positions[:, 1] = coords
    elif axis == 'z':
        positions[:, 2] = coords
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")
    return positions

def compute_virtual_antennas(T, R):
    N, _ = T.shape
    M, _ = R.shape
    T_expanded = np.repeat(T, M, axis=0)      # (N*M, 3)
    R_tiled = np.tile(R, (N, 1))             # (N*M, 3)
    V = T_expanded + R_tiled                 # (N*M, 3)
    return V

def plot_antenna_positions(T, R, V):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(T[:, 0], T[:, 1], T[:, 2], c='red', marker='o', s=60, label='Tx (发射)')
    ax.scatter(R[:, 0], R[:, 1], R[:, 2], c='blue', marker='^', s=60, label='Rx (接收)')
    ax.scatter(V[:, 0], V[:, 1], V[:, 2], c='green', marker='x', s=40, label='Virtual (虚拟)')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('ULA 天线与虚拟阵列示意图')
    ax.legend()
    ax.grid(True)
    plt.show()

if __name__ == '__main__':
    wavelength = 1.0  # 单位: λ
    spacing = wavelength / 2

    N = 2  # 发射阵列数量
    M = 4  # 接收阵列数量

    T = create_ula(N, 4*spacing, axis='x')
    R = create_ula(M, spacing, axis='x') + np.array([8, 0, 0])  # 接收阵列平移

    # define 12 TX positions
    T = np.array(
        [
            [6, 0, 0],    # tx 0
            [8, 1, 0],    # tx 1
            [10, 0, 0],    # tx 1     
        ]
    )

    # define 16 RX positions
    R = np.array(
        [
            [0, 0, 0],    # rx 0
            [1, 0, 0],    # rx 1
            [2, 0, 0],    # rx 2
            [3, 0, 0],    # rx 3
        ]
    ) 


    print("发射阵列位置:\n", T)
    print("接收阵列位置:\n", R)

    V = compute_virtual_antennas(T, R)
    print(f"虚拟天线数量: {V.shape[0]} (应为 {N}×{M} = {N*M})")

    plot_antenna_positions(T, R, V)
