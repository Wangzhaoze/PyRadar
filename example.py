import matplotlib.pyplot as plt

from pyradar_sandbox.rsp.cfar import *
from pyradar_sandbox.rsp.fft import *
from scipy.io import loadmat

adc_path = "C:/Project_HOME/pyradar/examples/data/CRUW/2019_04_30_mlms001/radar_raw_frame/000015.mat"

data_cube = loadmat(adc_path)['adcData']

data_cube = data_cube.reshape((data_cube.shape[0], data_cube.shape[1], -1))  # reshape to (num_chirps, num_samples, num_antennas)

rfft = range_fft(data_cube, numRangeBins=128, IdxSamples=0, window_type='Blackman')
dfft = doppler_fft(rfft, numDopplerBins=255, IdxChirps=1, window_type='Blackman')

rd_map = np.abs(dfft[..., 0])

plt.figure(figsize=(16, 18))

# ================= 原始信号 =================
plt.subplot(5, 2, 1)
plt.imshow(rd_map, aspect='auto', cmap='jet')
plt.title("Range-Doppler Map")
plt.colorbar()

# ================= CFAR 可视化 =================
methods = [
    ("CA-CFAR", ca_cfar_2d(rd_map)),
    ("VI-CFAR", vi_cfar_2d(rd_map)),
    ("OA-CFAR", oa_cfar_2d(rd_map)),
    ("OS-CFAR", os_cfar_2d(rd_map)),
    ("TM-CFAR", trimmed_mean_cfar_2d(rd_map)),
    ("CML-CFAR", cml_cfar_2d(rd_map)),
    ("GOCA-CFAR", goca_cfar_2d(rd_map)),
    ("SOCA-CFAR", soca_cfar_2d(rd_map)),
    ("Median-CFAR", median_cfar_2d(rd_map)),
]

for idx, (name, mask) in enumerate(methods, start=2):
    plt.subplot(5, 2, idx)
    plt.imshow(mask, aspect='auto', cmap='jet')
    plt.title(name)
    plt.colorbar()

plt.tight_layout()
plt.show()

# from scipy.signal import stft, cwt, morlet2

# # ================= Micro-Doppler 分析 =================

# # 1. 选择一个 range bin（假设检测到目标在 range_bin=30 附近）
# range_bin = 150

# # 从 rfft 中取对应区间的时序信号 (num_chirps, num_antennas)
# target_signal = rfft[:, range_bin, 0]   # shape: (num_chirps, num_bins)

# # 2. 短时傅里叶变换 (STFT)
# f, t_stft, Zxx = stft(target_signal, fs=1.0, nperseg=128, noverlap=64)

# plt.figure(figsize=(12, 5))
# plt.pcolormesh(t_stft, f, np.abs(Zxx), shading='gouraud')
# plt.title(f"Micro-Doppler via STFT (Range bin {range_bin})")
# plt.ylabel("Doppler Frequency Bin")
# plt.xlabel("Slow Time (Chirps)")
# plt.colorbar(label="Amplitude")
# plt.show()

# # 3. 小波变换 (CWT, 用 Morlet 小波替代 pywt)
# widths = np.arange(1, 64)  # 类似 scales
# cwt_matrix = cwt(target_signal, morlet2, widths, w=5.0)

# plt.figure(figsize=(12, 5))
# plt.pcolormesh(np.arange(len(target_signal)), widths, np.abs(cwt_matrix), shading='gouraud')
# plt.title(f"Micro-Doppler via CWT (Range bin {range_bin})")
# plt.ylabel("Scale")
# plt.xlabel("Slow Time (Chirps)")
# plt.colorbar(label="Magnitude")
# plt.show()


# from scipy.signal import stft, cwt, morlet2

# # 假设通过 CFAR 得到目标在 range_bin 区间 [28, 32]
# range_bins = np.arange(144, 150)

# # 从 rfft 中取对应区间的时序信号 (num_chirps, num_antennas)
# signals = rfft[:, range_bins, 0]   # shape: (num_chirps, num_bins)

# # 非相干合并：能量相加
# target_signal = np.sqrt(np.sum(np.abs(signals)**2, axis=1))

# # 如果你想尝试相干合并（注意相位对齐问题）
# # target_signal = np.sum(signals, axis=1)

# # ================== STFT ==================
# f, t_stft, Zxx = stft(target_signal, fs=1.0, nperseg=64, noverlap=32)

# plt.figure(figsize=(12, 5))
# plt.pcolormesh(t_stft, f, np.abs(Zxx), shading='gouraud')
# plt.title("Micro-Doppler via STFT (Merged Range Bins 28-32)")
# plt.ylabel("Doppler Frequency Bin")
# plt.xlabel("Slow Time (Chirps)")
# plt.colorbar(label="Amplitude")
# plt.show()

# # ================== CWT ==================
# from scipy.signal import cwt, morlet2
# widths = np.arange(1, 64)
# cwt_matrix = cwt(target_signal, morlet2, widths, w=5.0)

# plt.figure(figsize=(12, 5))
# plt.pcolormesh(np.arange(len(target_signal)), widths, np.abs(cwt_matrix), shading='gouraud')
# plt.title("Micro-Doppler via CWT (Merged Range Bins 28-32)")
# plt.ylabel("Scale")
# plt.xlabel("Slow Time (Chirps)")
# plt.colorbar(label="Magnitude")
# plt.show()
