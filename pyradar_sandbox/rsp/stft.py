import numpy as np
import scipy.io as sio
import scipy.signal as signal
import matplotlib.pyplot as plt

# ----------------------------
# 参数初始化
# ----------------------------
def get_params_value():
    c = 3e8  # 光速
    fc = 77e9
    lam = c / fc
    Rx, Tx = 4, 2

    Fs = 4e6
    sweepSlope = 21.0017e12
    samples = 128
    loop = 255
    Tc = 120e-6
    fft_Rang = 134
    fft_Vel = 256
    fft_Ang = 128
    num_crop = 3
    max_value = 1e4

    # range grid
    freq_res = Fs / fft_Rang
    freq_grid = np.arange(fft_Rang)[:, None] * freq_res
    rng_grid = (freq_grid * c / sweepSlope / 2).flatten()

    # angle grid（简化：-1..1 的空间频率映射到 arcsin）
    w = np.linspace(-1, 1, fft_Ang)
    agl_grid = np.arcsin(np.clip(w, -1, 1)) * 180 / np.pi

    # velocity grid（Doppler 采样率按 1/Tc，与 MATLAB 保持一致）
    fs_dop = 1.0 / Tc
    dop_grid = np.fft.fftshift(np.fft.fftfreq(fft_Vel, d=1.0 / fs_dop))
    vel_grid = dop_grid * lam / 2

    return {
        'c': c, 'fc': fc, 'lambda': lam, 'Rx': Rx, 'Tx': Tx,
        'Fs': Fs, 'sweepSlope': sweepSlope, 'samples': samples, 'loop': loop,
        'Tc': Tc, 'fft_Rang': fft_Rang, 'fft_Vel': fft_Vel, 'fft_Ang': fft_Ang,
        'num_crop': num_crop, 'max_value': max_value,
        'rng_grid': rng_grid, 'agl_grid': agl_grid, 'vel_grid': vel_grid
    }

# ----------------------------
# 基础 FFT 与工具函数
# ----------------------------
def fft_range(chirp_data, n_fft, window=True):
    x = chirp_data
    if window:
        w = np.hanning(x.shape[0])[:, None, None]
        x = x * w
    X = np.fft.fft(x, n=n_fft, axis=0)
    return X

def fft_doppler(range_cube, n_fft):
    X = np.fft.fft(range_cube, n=n_fft, axis=2)
    X = np.fft.fftshift(X, axes=2)
    return X

def fft_angle(merged_cube, n_fft):
    X = np.fft.fft(merged_cube, n=n_fft, axis=1)
    X = np.fft.fftshift(X, axes=1)
    return X

def Normalize(x, max_value):
    mag = np.abs(x)
    mag = np.clip(mag, 0, max_value) / (max_value + 1e-12)
    return mag

# ----------------------------
# CFAR 检测 + 聚峰
# ----------------------------
def cfar_RV(rv_map, fft_Rang, num_crop, Pfa,
            guard_r=2, train_r=8, guard_v=2, train_v=8):
    R, V = rv_map.shape
    dets = []
    Tr = (2*train_r + 2*guard_r + 1)
    Tv = (2*train_v + 2*guard_v + 1)
    total_cells = Tr*Tv - (2*guard_r + 1)*(2*guard_v + 1)
    alpha = total_cells * (Pfa ** (-1.0 / total_cells) - 1.0)
    r_min = num_crop + train_r + guard_r
    r_max = R - (num_crop + train_r + guard_r)
    v_min = train_v + guard_v
    v_max = V - (train_v + guard_v)
    for r in range(r_min, r_max):
        for v in range(v_min, v_max):
            r0, r1 = r - (train_r + guard_r), r + (train_r + guard_r) + 1
            v0, v1 = v - (train_v + guard_v), v + (train_v + guard_v) + 1
            window = rv_map[r0:r1, v0:v1].copy()
            gr0, gr1 = r - guard_r - r0, r + guard_r + 1 - r0
            gv0, gv1 = v - guard_v - v0, v + guard_v + 1 - v0
            window[gr0:gr1, gv0:gv1] = 0.0
            mu = np.sum(window) / (total_cells + 1e-12)
            th = alpha * mu
            cut = rv_map[r, v]
            if cut > th and cut > 0:
                dets.append([v + 1, r + 1, cut])
    return np.array(dets) if len(dets) > 0 else np.empty((0, 3))

def peakGrouping(dets, nms_win_r=1, nms_win_v=1):
    if dets.size == 0: return dets
    dets_list = dets.tolist()
    dets_list.sort(key=lambda x: x[2], reverse=True)
    kept, occupied = [], set()
    for v_idx, r_idx, p in dets_list:
        taken = False
        for dr in range(-nms_win_r, nms_win_r + 1):
            for dv in range(-nms_win_v, nms_win_v + 1):
                if (r_idx + dr, v_idx + dv) in occupied:
                    taken = True
                    break
            if taken: break
        if not taken:
            kept.append([v_idx, r_idx, p])
            occupied.add((r_idx, v_idx))
    return np.array(kept) if len(kept) > 0 else np.empty((0, 3))

# ----------------------------
# 角度估计
# ----------------------------
def angle_estim_dets(detout, dopplerdata_merge, fft_Vel, fft_Ang, Rx, Tx, num_crop):
    if detout.size == 0:
        return [], None, []
    R, VIRT, V = dopplerdata_merge.shape
    resel_agl, rng_excd = [], []
    for i in range(detout.shape[0]):
        v_idx_1b = int(detout[i, 0])
        r_idx_1b = int(detout[i, 1])
        if r_idx_1b <= num_crop or r_idx_1b > (R - num_crop):
            rng_excd.append(i)
            resel_agl.append(1)
            continue
        r0, v0 = r_idx_1b - 1, v_idx_1b - 1
        rv_vec = dopplerdata_merge[r0, :, v0]
        ang_spec = np.fft.fft(rv_vec, n=fft_Ang, axis=0)
        ang_spec = np.fft.fftshift(ang_spec, axes=0)
        a_idx = int(np.argmax(np.abs(ang_spec))) + 1
        resel_agl.append(a_idx)
    return resel_agl, None, rng_excd

# ----------------------------
# 简易聚类
# ----------------------------
def clustering(save_det_data, fft_Vel, veloc_bin_norm, dis_thrs, rng_grid, agl_grid):
    if save_det_data.size == 0:
        return np.empty((0, 3))
    rng_th, vel_th, ang_th = dis_thrs
    rng_bin = save_det_data[:, 0].astype(int)
    vel_bin = save_det_data[:, 1].astype(int)
    ang_bin = save_det_data[:, 2].astype(int)
    pow_w   = save_det_data[:, 3].astype(float)
    vel_bin_n = vel_bin / float(veloc_bin_norm)
    N = len(rng_bin)
    visited = np.zeros(N, dtype=bool)
    clusters = []
    for i in range(N):
        if visited[i]: continue
        queue = [i]
        visited[i] = True
        cls = [i]
        while queue:
            q = queue.pop(0)
            dr = np.abs(rng_bin - rng_bin[q])
            dv = np.abs(vel_bin_n - vel_bin_n[q])
            da = np.abs(ang_bin - ang_bin[q])
            neigh = np.where((dr <= rng_th) & (dv <= vel_th) & (da <= ang_th) & (~visited))[0]
            for nb in neigh:
                visited[nb] = True
                queue.append(nb)
                cls.append(nb)
        clusters.append(cls)
    centers = []
    for idxs in clusters:
        w = pow_w[idxs]
        rb = rng_bin[idxs]
        vb = vel_bin[idxs]
        ab = ang_bin[idxs]
        def w_median(vals, weights):
            order = np.argsort(vals)
            v = vals[order]
            w_sorted = weights[order]
            csum = np.cumsum(w_sorted) / (np.sum(w_sorted) + 1e-12)
            pos = np.searchsorted(csum, 0.5)
            return v[min(pos, len(v)-1)]
        cr = int(np.round(w_median(rb, w)))
        cv = int(np.round(w_median(vb, w)))
        ca = int(np.round(w_median(ab, w)))
        centers.append([cr, cv, ca, np.sum(w)])
    centers = np.array(centers)
    centers = centers[np.argsort(-centers[:, 3])]
    return centers[:, :3].astype(int)

# ----------------------------
# 主流程
# ----------------------------
def main():
    params = get_params_value()
    Rx, Tx = params['Rx'], params['Tx']
    samples = params['samples']
    loop = params['loop']
    fft_Rang = params['fft_Rang']
    fft_Vel = params['fft_Vel']
    fft_Ang = params['fft_Ang']
    num_crop = params['num_crop']
    max_value = params['max_value']
    rng_grid = params['rng_grid']
    agl_grid = params['agl_grid']
    vel_grid = params['vel_grid']
    lam = params['lambda']
    Tc = params['Tc']

    # ROI & STFT 参数
    Lr, La = 11, 5
    Ang_seq = [2, 5, 8, 11, 14]
    veloc_bin_norm = 2
    dis_thrs = [20, 16, 20]
    WINDOW, NOVERLAP, NFFT = 255, 240, 256

    # 读取 4D 数据
    # 多帧累积
    radarcube_crop_all = []
    for frame_idx in range(10, 30):

        seq_dir = f'C:/Users/wangzh179/Desktop/Project_TERRA/GenRAD/data/2019_04_09_bms1000/radar_raw_frame/0000{frame_idx}.mat'
        mat = sio.loadmat(seq_dir)
        data_frames = mat['adcData']  # (samples, loop, Rx, Tx)
        assert data_frames.shape == (samples, loop, Rx, Tx), f"adcData shape mismatch: {data_frames.shape}"

        # --- 单帧处理，TDM-MIMO 奇/偶分离 ---
        chirp_odd  = np.transpose(data_frames[:, :, :, 0], (0, 2, 1))
        chirp_even = np.transpose(data_frames[:, :, :, 1], (0, 2, 1))
        Rangedata_odd  = fft_range(chirp_odd,  fft_Rang, window=True)
        Rangedata_even = fft_range(chirp_even, fft_Rang, window=True)
        Dopplerdata_odd  = fft_doppler(Rangedata_odd,  fft_Vel)
        Dopplerdata_even = fft_doppler(Rangedata_even, fft_Vel)
        Dopdata_sum = np.mean(np.abs(Dopplerdata_odd), axis=1)
        Pfa = 1e-4
        Resl_indx = cfar_RV(Dopdata_sum, fft_Rang, num_crop, Pfa)
        detout = peakGrouping(Resl_indx)

        # 多普勒相位补偿
        if detout.size != 0:
            for r_1b in range(num_crop + 1, fft_Rang - num_crop + 1):
                idxs = np.where(detout[:, 1] == r_1b)[0]
                if idxs.size == 0: continue
                k_1b = detout[idxs[0], 0]
                pha_comp_term = np.exp(-1j * np.pi * (k_1b - fft_Vel/2.0 - 1.0) / fft_Vel)
                Rangedata_even[r_1b-1, :, :] *= pha_comp_term

        # 合并 TX + 角度 FFT
        Rangedata_merge = np.concatenate((Rangedata_odd, Rangedata_even), axis=1)
        Angdata = fft_angle(Rangedata_merge, fft_Ang)
        Angdata_crop = Normalize(Angdata[num_crop:fft_Rang - num_crop, :, :], max_value)

        # 裁剪中心
        if detout.size != 0:
            Doppler_merge = np.concatenate((Dopplerdata_odd, Dopplerdata_even), axis=1)
            Resel_agl, _, rng_excd_list = angle_estim_dets(detout, Doppler_merge, fft_Vel, fft_Ang, Rx, Tx, num_crop)
            rng_bins = detout[:, 1].astype(int)
            vel_bins = detout[:, 0].astype(int)
            ang_bins = np.array(Resel_agl).astype(int)
            power_vals = detout[:, 2]
            rng_m = rng_grid[rng_bins - 1]
            vel_m = vel_grid[vel_bins - 1]
            ang_deg = agl_grid[ang_bins - 1]
            save_det_data = np.column_stack([rng_bins, vel_bins, ang_bins, power_vals, rng_m, vel_m, ang_deg])
            if len(rng_excd_list) > 0:
                mask = np.ones(save_det_data.shape[0], dtype=bool)
                mask[np.array(rng_excd_list, dtype=int)] = False
                save_det_data = save_det_data[mask]
            dets_cluster = clustering(save_det_data, fft_Vel, veloc_bin_norm, dis_thrs, rng_grid, agl_grid)
            if dets_cluster.size > 0:
                center_r = int(dets_cluster[0, 0])
                center_a = int(dets_cluster[0, 2])
            else:
                center_r = int(round(fft_Rang / 2))
                center_a = int(round(fft_Ang / 2))
        else:
            center_r = int(round(fft_Rang / 2))
            center_a = int(round(fft_Ang / 2))

        # 按中心裁剪 Lr × La × T
        half_Lr = (Lr - 1) // 2
        r_center0_in_crop = (center_r - 1) - num_crop
        r_start = max(r_center0_in_crop - half_Lr, 0)
        r_end   = min(r_start + Lr, Angdata_crop.shape[0])
        a_center0 = center_a - 1
        a_idxs_rel = a_center0 - (max(Ang_seq)//2 + 1) + np.array(Ang_seq, dtype=int)
        a_idxs0 = np.clip(a_idxs_rel, 0, fft_Ang - 1)
        cropped = Angdata_crop[r_start:r_end, a_idxs0, :]
        if cropped.shape[0] < Lr:
            pad_rows = Lr - cropped.shape[0]
            cropped = np.pad(cropped, ((0, pad_rows), (0, 0), (0, 0)), mode='constant', constant_values=0)
        radarcube_crop_all.append(cropped)  


    radarcube_crop = np.concatenate(radarcube_crop_all, axis=2)
    Lr_, La_, T_ = radarcube_crop.shape
    data_conca = radarcube_crop.reshape(Lr_*La_, T_)

    # STFT
    STFT_data = []
    for h in range(Lr_ * La_):
        x = data_conca[h, :]
        f, t, Sxx = signal.spectrogram(
            x,
            fs=1/Tc,
            window='hann',
            nperseg=WINDOW,
            noverlap=NOVERLAP,
            nfft=NFFT,
            detrend=False,
            return_onesided=False,
            mode='complex',
            scaling='density'
        )
        Sxx = np.fft.fftshift(Sxx, axes=0)
        STFT_data.append(Sxx)
    STFT_data = np.stack(STFT_data, axis=-1)
    v_grid_new = np.fft.fftshift(f) * lam / 2.0

    # 绘图
    mid_idx = (Lr_ * La_) // 2
    plt.figure(figsize=(7.5, 4.5))
    plt.imshow(np.abs(STFT_data[:, :, mid_idx]), aspect='auto', origin='lower',
               extent=[t[0], t[-1], v_grid_new[0], v_grid_new[-1]], cmap='jet')
    plt.colorbar()
    plt.xlabel('time (s)')
    plt.ylabel('velocity (m/s)')
    plt.title('micro-Doppler map (center ROI pixel)')
    plt.tight_layout()
    plt.show()
    print("完成：radarcube_crop 形状 =", radarcube_crop.shape)

if __name__ == "__main__":
    main()
