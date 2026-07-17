# CFAR detection

CFAR consumes nonnegative power, not complex amplitudes. It returns four full-size
arrays: a boolean detection mask, estimated noise, threshold, and SNR in decibels.

## CA-CFAR

For $N$ exponentially distributed training cells, CA-CFAR estimates their mean
$\hat{P}_n$ and uses threshold

$$
T=\alpha\hat{P}_n,
\qquad
\alpha=N\left(P_\mathrm{FA}^{-1/N}-1\right).
$$

This relation gives the configured false-alarm probability under independent,
homogeneous complex Gaussian noise.

## GOCA and SOCA

Greatest-of and smallest-of CFAR split the training region into leading and
trailing halves:

$$
\hat{P}_\mathrm{GOCA}=\max(\hat{P}_L,\hat{P}_R),
\qquad
\hat{P}_\mathrm{SOCA}=\min(\hat{P}_L,\hat{P}_R).
$$

GOCA is conservative at clutter boundaries. SOCA can preserve a target beside a
high-clutter region but may increase false alarms.

## OS-CFAR

OS-CFAR sorts the $N$ training powers and chooses rank $k$. The multiplier is
solved from the exact Laplace transform of the exponential order statistic:

$$
P_\mathrm{FA}=\prod_{i=0}^{k}
\frac{N-i}{N-i+\alpha}.
$$

`rankFraction` selects $k$ and is commonly set near 0.75. OS-CFAR is useful when
other targets contaminate some training cells.

## Two-dimensional behavior

`cfar_2d` treats Doppler as circular and range as bounded. Range edge cells whose
full training window is unavailable are invalid and receive `NaN` noise,
threshold, and SNR. Optional 3x3 peak grouping performs deterministic NMS.

```python
result = rsp.cfar_2d(
    rdPower,
    method="os",
    trainingCells=(10, 4),
    guardCells=(2, 1),
    rankFraction=0.75,
    pfa=1e-4,
    peakGrouping=True,
)
bins = np.argwhere(result.detections)
```

There is no top-N fallback when no cell passes CFAR. An empty detection set is a
valid result and must remain empty. `maxDetections` only caps genuine detections
after sorting by SNR in the model pipeline.

## Statistical validation

An empirical PFA test should disable peak grouping, draw independent exponential
power, discard invalid range edges, and compare the observed false-alarm count to
a binomial confidence interval around the configured PFA. Structured clutter and
correlated FFT bins do not satisfy the ideal CA-CFAR assumptions.
