# Detection-first point clouds

The default 4D pipeline detects on range-Doppler power before estimating angle.
This avoids allocating a dense range-Doppler-azimuth-elevation tensor when only a
small number of cells contain targets.

## RD power and detections

After MIMO timing compensation and array calibration,

$$
P[r,d]=\frac{1}{M}\sum_{m=0}^{M-1}|X[r,d,m]|^2.
$$

CFAR produces range and Doppler source bins. For each surviving cell, nearby
range/Doppler samples provide snapshots for the selected DoA method.

## Coordinate conversion

The physical values are

$$
R = k_r\Delta R_\mathrm{bin},
\qquad
v_r = \mathrm{velocityAxis}[k_d].
$$

With azimuth $\theta$ and elevation $\phi$, FLU coordinates are

$$
x=R\cos\phi\cos\theta,\qquad
y=R\cos\phi\sin\theta,\qquad
z=R\sin\phi.
$$

`PointCloud` stores `xyz`, `power`, `snr`, `radialVelocity`, timestamp, frame id,
and source bins `(range, doppler, azimuth, elevation)`. `to_numpy()` returns
columns `x, y, z, power, snr, radialVelocity`; provenance remains available on
the typed object.

## Dense research products

`retainRangeDoppler`, `retainRangeAngle`, and `retainRangeDopplerAngle` control
intermediate products. The default RA visualization beamforms the strongest
Doppler cell at each range. Full RDA is generated only when explicitly requested.
Elevation remains available per detection even when a dense azimuth-only product
is retained.

## Ambiguities

Point coordinates are estimates inside the model's unambiguous range and angular
FOV. TDM profiles with calibrated overlap phase centres can resolve velocity
aliases inside the raw chirp-rate Nyquist interval; the selected order is stored
in frame metadata. Without that evidence the library preserves the aliased
velocity rather than claiming a unique physical target. Application-specific
range unwrapping remains outside v1.
