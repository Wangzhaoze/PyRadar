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