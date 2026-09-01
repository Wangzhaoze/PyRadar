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
:hidden:

getting-started
tutorials/index
auto_examples/index
api/index
project
```

## Documentation

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Getting Started
:link: getting-started
:link-type: doc

Install `pyradar`, define a radar model, process ADC data, and understand the
core data conventions.
:::

:::{grid-item-card} User Guide
:link: tutorials/index
:link-type: doc

Learn FMCW processing, MIMO and DoA, CFAR, point-cloud generation, tracking,
simulation, and time-frequency analysis.
:::

:::{grid-item-card} Examples
:link: auto_examples/index
:link-type: doc

Browse executable end-to-end examples and dataset-oriented workflows.
:::

:::{grid-item-card} API Reference
:link: api/index
:link-type: doc

Look up radar-model contracts, signal-processing functions, simulation APIs,
and utility adapters.
:::

:::{grid-item-card} Project
:link: project
:link-type: doc

Read release notes and contributor-facing development documentation.
:::

::::

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
