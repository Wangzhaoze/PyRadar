# FFT stages and windows

## Named input dimensions

Raw ADC enters as `loop/emission/rx/sample`. Range FFT acts on `sample`; MIMO
decoding converts `emission/rx` into `virtual`; Doppler FFT then acts on `loop`.

```text
(loop, emission, rx, sample)
             range FFT |
                       v
(loop, emission, rx, range)
             MIMO decode
                       v
(loop, virtual, range)
           Doppler FFT |
                       v
(range, doppler, virtual)
```

For TDM this order is important: a loop means a complete TX cycle, while
`emission` identifies a chirp within that cycle.

## Range transform

For window $w[n]$ and optional mean removal,

$$
X_r[k]=\sum_{n=0}^{N_s-1} w[n](x[n]-\bar{x})
       \exp\left(-j\frac{2\pi kn}{N_R}\right).
$$

`removeRangeMean` suppresses a sample-independent DC component. It is not a
substitute for measured coupling subtraction.

```python
rangeCube = rsp.range_fft(adcFrame, radar=radar)
```

The parameter form does not require a model:

```python
spectrum = rsp.range_fft(
    adc,
    fftSize=512,
    sampleAxis=-1,
    window="blackmanharris",
    removeMean=True,
)
```

## Doppler transform

After MIMO decoding, slow-time FFT is shifted so zero radial velocity is in the
centre:

$$
X_D[k]=\operatorname{fftshift}\left\{
\sum_{\ell=0}^{N_L-1} w_D[\ell]X_r[\ell]
\exp\left(-j\frac{2\pi k\ell}{N_D}\right)
\right\}.
$$

`removeDopplerMean` subtracts the slow-time mean and suppresses stationary
clutter. Disable it when preserving zero-Doppler reflectors is important.

## Choosing a window

| Window | Main-lobe width | Sidelobe suppression | Typical use |
| --- | --- | --- | --- |
| rectangular | narrowest | poor | coherent synthetic tests |
| Hann | moderate | good | general range/Doppler processing |
| Hamming | moderate | good first sidelobe | general processing |
| Blackman | wider | stronger | high dynamic range |
| Blackman-Harris | widest | strongest of these | weak target beside strong target |

Windowing trades resolution for leakage suppression. `pyradar` does not silently
renormalize coherent gain: absolute power calibration must include the selected
window and FFT convention. Relative peak locations and CFAR inputs remain
consistent within one configured pipeline.

## Cropping

`FFTConfig.rangeCrop`, `azimuthCrop`, and `elevationCrop` use Python slice
semantics. Cropping changes retained bins, not the physical bin spacing. The
source range-bin indices remain in cube metadata and point-cloud provenance.
