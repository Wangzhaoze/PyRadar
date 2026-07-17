# pyradar

`pyradar` processes sawtooth FMCW ADC samples using an explicit model of the
radar that captured them. Array positions, emission timing, waveform physics,
sampling, calibration, and algorithm defaults travel together as one immutable
{class}`pyradar.base.Radar`. This lets the same processing functions operate on
a SIMO ULA, an AWR1843, or a sparse 192-channel cascade without dataset branches.

The v1 pipeline follows the physical data path:

```text
ADC calibration -> range FFT -> MIMO decode -> Doppler FFT -> RD CFAR
                -> emission-time phase compensation -> DoA -> point cloud
                -> clustering -> multi-target tracking
```

All public values use SI units and radians. Coordinates are right-handed FLU:
`x` forward, `y` left, and `z` up.

```{toctree}
:maxdepth: 2
:caption: Start here

getting-started
radar-model
data-and-licenses
auto_examples/index
```

```{toctree}
:maxdepth: 2
:caption: Algorithm tutorials

tutorials/fmcw
tutorials/fft-windows
tutorials/mimo
tutorials/virtual-array-doa
tutorials/cfar
tutorials/calibration
tutorials/pointcloud
tutorials/clustering
tutorials/tracking
```

```{toctree}
:maxdepth: 2
:caption: Reference

api/index
release-notes
development/release
development/native-acceleration
```

## Two API levels

Use the model-aware API for production captures:

```python
result = radar.process_adc(
    adc,
    dims=("loop", "emission", "rx", "sample"),
)
points = result.pointCloud
```

Use parameter functions when studying one transform in isolation:

```python
from pyradar import rsp

spectrum = rsp.range_fft(
    adc,
    fftSize=512,
    sampleAxis=-1,
    window="hann",
)
```

Explicit function arguments override model defaults. Raw arrays never receive
guessed dimension names; ambiguous input is rejected before any FFT is run.
