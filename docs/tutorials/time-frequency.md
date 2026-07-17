# Time-frequency and preprocessing

Range-Doppler processing assumes target Doppler is nearly stationary over one
coherent processing interval.  Micro-motion violates that assumption in useful
ways: rotating wheels, limbs, and vibration create time-varying Doppler.  The
short-time Fourier transform (STFT) estimates a local spectrum,

$$
X(m,f)=\sum_n x[n]w[n-mH]e^{-j2\pi fn/F_s},
$$

where $w$ is a window and $H$ is the hop size.  Smaller segments improve time
localization; longer segments improve Doppler resolution.  Overlap changes the
sampling of time, not the information bandwidth.

```python
from pyradar import rsp

spectrogram = rsp.micro_doppler_spectrogram(
    slowTimeSignal,
    radar=radar,
    segmentLength=128,
    overlap=112,
    fftSize=512,
    window="hann",
)

power = spectrogram.power
velocity = spectrogram.velocity  # m/s, positive approaching
time = spectrogram.time           # s
```

{func}`pyradar.rsp.stft` is the parameter-only form and returns two-sided
frequency coordinates.  {func}`pyradar.rsp.micro_doppler_spectrogram` derives
the slow-time sample rate and velocity coordinate from `Radar` when supplied.

## Zoom FFT

Zero-padding samples a conventional DFT more densely but still computes the
entire Nyquist interval.  {func}`pyradar.rsp.zoom_fft` uses the chirp z-transform
to evaluate only a requested frequency interval:

```python
zoom = rsp.zoom_fft(
    adc,
    frequencyRange=(120e3, 180e3),
    sampleRate=radar.sampler.sampleRate,
    fftSize=2048,
    axis=-1,
)
```

The interval is half-open by default, so adjacent intervals do not duplicate an
endpoint.  Zoom FFT improves frequency-grid spacing in the selected band; it
does not surpass the waveform's physical range resolution.

## Composable preprocessing

`remove_static_clutter` subtracts the coherent slow-time mean and therefore
places a notch at zero Doppler.  It should be disabled when stationary targets
matter or the coherent interval is too short.

`coherent_integrate` sums complex samples and preserves phase gain.  It requires
phase alignment.  `noncoherent_integrate` sums magnitudes or powers and is safer
across incoherent channels at the cost of coherent processing gain.  These are
parameter functions; pipeline defaults remain in
{class}`~pyradar.base.ProcessingConfig`.
