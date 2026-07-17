# pyradar

`pyradar` is a radar-model-centric Python library for turning sawtooth FMCW ADC
samples into range-Doppler products, detections, velocity-bearing point clouds,
clusters, and tracks. The radar model owns waveform, sampling, array geometry,
MIMO timing, calibration, and processing defaults, so dataset-specific logic does
not leak into signal-processing functions.

```python
from pyradar.base import Radar

radar = Radar.from_config("radar.yaml")
result = radar.build_pipeline().process(
    adc,
    dims=("loop", "emission", "rx", "sample"),
)

points = result.pointCloud.to_numpy()  # x, y, z, power, snr, radialVelocity
tracks = result.tracks
```

The lower-level parameter API remains available for experiments:

```python
from pyradar import rsp

rangeSpectrum = rsp.range_fft(
    adc,
    fftSize=512,
    sampleAxis=-1,
    window="hann",
)
rangeDoppler = rsp.range_doppler_fft(adc, radar=radar, dims=dims)
```

## Scope

- SIMO, arbitrary-order TDM, two-channel BPM, and phase-coded DDM
- ULA, complete rectangular, sparse, and duplicate-phase-center arrays
- Range, Doppler, and angle FFT; Bartlett, Capon, MUSIC, and ESPRIT DoA
- CA-, GOCA-, SOCA-, and OS-CFAR with noise, threshold, SNR, and NMS outputs
- FLU point clouds in SI units, deterministic velocity-aware DBSCAN, and
  constant-velocity 2D/3D multi-target tracking
- Readers for ColoRadar, ColoRadar+, RaDelft, and RAMPCNN/CRUW captures

## Installation

Install a wheel downloaded from the GitHub Releases page:

```bash
python -m pip install pyradar-1.0.0rc1-py3-none-any.whl
```

For development:

```bash
python -m pip install -e ".[test,docs,examples]"
python -m pytest
python -m build
```

The distribution is intentionally not uploaded to PyPI because the `pyradar`
distribution name is already used by another project. The import name remains
`pyradar`.

## Documentation

The documentation site contains the model contract, equations, algorithm
assumptions, dataset adapters, executable examples, and generated API reference.
Build it locally with:

```bash
python -m sphinx -W -b html docs docs/_build/html
```

All public calculations use SI units, radians, and a right-handed FLU coordinate
frame: `x` forward, `y` left, `z` up.

## Data and license

The source code is released under the MIT license. Demo data retains its original
dataset license; see `docs/examples/data/DATA.md` and `manifest.json`. RaDelft,
ColoRadar+, and RAMPCNN/CRUW examples use local paths and do not redistribute
those datasets.
