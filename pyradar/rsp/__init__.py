"""Radar signal-processing algorithms."""

from .cfar import (
    CFARResult,
    ca_cfar_1d,
    ca_cfar_2d,
    cfar_1d,
    cfar_2d,
    goca_cfar_1d,
    goca_cfar_2d,
    os_cfar_1d,
    os_cfar_2d,
    soca_cfar_1d,
    soca_cfar_2d,
)
from .clustering import dbscan
from .doa import (
    DoAResult,
    combine_duplicate_channels,
    doa_bartlett,
    doa_capon,
    doa_esprit,
    doa_music,
    estimate_doa,
    spatial_covariance,
    spatial_smoothing,
    steering_vector,
)
from .fft import (
    angle_fft,
    doppler_fft,
    range_doppler_azimuth_fft,
    range_doppler_fft,
    range_fft,
    window,
)
from .mimo import DopplerUnwrapResult, unwrap_tdm_velocity
from .pipeline import ADCToPointCloudPipeline
from .pointcloud import detections_to_pointcloud, spherical_to_cartesian
from .preprocessing import (
    coherent_integrate,
    noncoherent_integrate,
    remove_static_clutter,
)
from .time_frequency import (
    TimeFrequencyResult,
    ZoomFFTResult,
    micro_doppler_spectrogram,
    stft,
    zoom_fft,
)
from .tracking import MultiTargetTracker

__all__ = [
    "ADCToPointCloudPipeline",
    "CFARResult",
    "DoAResult",
    "DopplerUnwrapResult",
    "MultiTargetTracker",
    "TimeFrequencyResult",
    "ZoomFFTResult",
    "angle_fft",
    "ca_cfar_1d",
    "ca_cfar_2d",
    "cfar_1d",
    "cfar_2d",
    "combine_duplicate_channels",
    "coherent_integrate",
    "dbscan",
    "detections_to_pointcloud",
    "doa_bartlett",
    "doa_capon",
    "doa_esprit",
    "doa_music",
    "doppler_fft",
    "estimate_doa",
    "goca_cfar_1d",
    "goca_cfar_2d",
    "micro_doppler_spectrogram",
    "noncoherent_integrate",
    "os_cfar_1d",
    "os_cfar_2d",
    "range_doppler_azimuth_fft",
    "range_doppler_fft",
    "range_fft",
    "remove_static_clutter",
    "soca_cfar_1d",
    "soca_cfar_2d",
    "spatial_covariance",
    "spatial_smoothing",
    "spherical_to_cartesian",
    "steering_vector",
    "stft",
    "unwrap_tdm_velocity",
    "window",
    "zoom_fft",
]
