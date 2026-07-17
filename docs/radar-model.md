# The radar model

## Why the model is the algorithm context

Range, Doppler, and angle are not properties of an ndarray. They depend on chirp
slope, sampling rate, emission timing, wavelength, and physical antenna phase
centres. `Radar` is therefore an immutable aggregate rather than a loose bag of
optional keyword arguments.

| Component | Responsibility |
| --- | --- |
| `FMCW` | sawtooth chirp timing and frequency law |
| `Sampler` | ADC shape, rate, bit depth, and frame period |
| `Transceivers` | TX/RX phase-centre coordinates in metres |
| `MIMOScheme` | channel map, emission order, decoding, timing correction |
| `Calibration` | ADC-, range-, and array-domain correction arrays |
| `ProcessingConfig` | layered algorithm defaults and retained products |

Instances are frozen dataclasses. Shape and physical consistency are validated at
construction, including ADC capture duration versus the chirp ramp and TX/RX
counts versus the MIMO strategy.

## Derived quantities

The following properties are calculated from the physical model:

- `rangeResolution`, `rangeBinSize`, and `maxUnambiguousRange`
- `velocityResolution`, `velocityBinSize`, and `maxUnambiguousVelocity`
- `virtualArray`, `duplicatePhaseCenters`, and `arrayGeometry`
- `unambiguousFov`, `rangeAxis`, `velocityAxis`, `azimuthAxis`, and
  `elevationAxis`

Resolution and FFT-bin spacing are intentionally distinct. Zero padding can make
`rangeBinSize` smaller, but cannot improve `rangeResolution`.

## Hardware and capture profiles

`TI2243CascadeRadar` represents the four-chip AWR2243 cascade. RaDelft,
ColoRadar, and ColoRadar+ are capture profiles that instantiate that hardware
with different waveform, sampler, TX order, and calibration metadata:

```python
from pyradar.base import TI2243CascadeRadar

radar = TI2243CascadeRadar.radelft()
```

`AWR1843Radar.rampcnn()` provides the RAMPCNN/CRUW profile. Dataset readers may
replace fallback profile values with parsed JSON, MAT, or text metadata. A profile
must not force a desired bin size by changing physical parameters.

## Processing configuration

Configuration is split by stage:

```python
from pyradar.base import (
    CFARConfig,
    DoAConfig,
    FFTConfig,
    ProcessingConfig,
)

processing = ProcessingConfig(
    fft=FFTConfig(
        rangeFftSize=512,
        dopplerFftSize=128,
        rangeWindow="hann",
        dopplerWindow="hann",
    ),
    cfar=CFARConfig(method="os", pfa=1e-4),
    doa=DoAConfig(method="auto"),
    retainRangeAngle=True,
)
```

The default pipeline is detection-first. Dense range-angle and
range-Doppler-angle cubes are opt-in because a 4D cascade cube can otherwise
consume several gigabytes.

## Data products

`ADCFrame` and `RadarCube` attach names to dimensions. `DetectionSet`,
`PointCloud`, `ClusterSet`, `TrackSet`, and `FrameResult` make downstream fields
explicit and preserve source bins. These contracts are preferable to ndarray
columns whose meanings change between scripts.
