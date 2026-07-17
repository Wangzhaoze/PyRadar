# Virtual arrays and direction of arrival

## FLU steering convention

For azimuth $\theta$ and elevation $\phi$, the FLU unit direction is

$$
\mathbf{u}(\theta,\phi)=
\begin{bmatrix}
\cos\phi\cos\theta &
\cos\phi\sin\theta &
\sin\phi
\end{bmatrix}^{\mathsf T}.
$$

At phase centre $\mathbf{p}_m$, `steering_vector` uses

$$
a_m(\theta,\phi)=
\exp\left(j\frac{2\pi}{\lambda}\mathbf{u}^{\mathsf T}\mathbf{p}_m\right).
$$

Positive azimuth points left and positive elevation points up.

## Automatic method selection

`estimate_doa(..., method="auto")` inspects unique virtual phase centres:

- uniform linear array: one-dimensional angle FFT;
- complete uniform rectangular array: two-dimensional angle FFT;
- sparse or arbitrary array: geometry-aware Bartlett grid search.

Explicit `bartlett`, `capon`, `music`, or `esprit` overrides this choice. ESPRIT
requires a ULA. FFT methods require complete uniformly spaced geometry; missing
positions are not filled with fictional measurements.

## Covariance methods

For snapshot matrix $\mathbf{X}$ with channels in columns,

$$
\hat{\mathbf{R}}=\frac{1}{N}\mathbf{X}^{\mathsf H}\mathbf{X}.
$$

Bartlett evaluates

$$
P_B(\theta,\phi)=\mathbf{a}^{\mathsf H}\hat{\mathbf{R}}\mathbf{a}.
$$

Capon/MVDR evaluates

$$
P_C(\theta,\phi)=
\frac{1}{\mathbf{a}^{\mathsf H}
(\hat{\mathbf{R}}+\delta\mathbf{I})^{-1}\mathbf{a}},
$$

with scale-relative diagonal loading. MUSIC separates the covariance eigenvectors
into signal and noise subspaces and evaluates

$$
P_M(\theta,\phi)=
\frac{1}{\lVert\mathbf{E}_n^{\mathsf H}\mathbf{a}\rVert_2^2}.
$$

`numSources` must be smaller than the number of unique channels. MUSIC and Capon
need enough statistically useful snapshots; a single RD sample is generally
better served by Bartlett or FFT.

## Coherent sources

`spatial_smoothing` averages overlapping ULA subarray covariances and optionally
applies forward-backward averaging. This can restore covariance rank for coherent
multipath sources at the cost of aperture.

## Duplicate phase centres

Cascade arrays often contain repeated virtual positions. `duplicatePolicy`
controls combination:

- `first`: preserve the first channel;
- `noncoherent`: RMS amplitude with first-channel phase;
- `coherent`: complex mean after array calibration.

Coherent averaging is meaningful only when channel phase calibration is valid.
The model exposes `duplicatePhaseCenters` so profile tests can verify the topology.

## Spatial aliasing and FOV

For uniform spacing $d$, an approximate unambiguous angular limit is

$$
|\theta| \leq \sin^{-1}\left(\min\left[1,\frac{\lambda}{2d}\right]\right).
$$

`Radar.unambiguousFov` derives this independently for the lateral and vertical
coordinates. A requested search FOV wider than this can contain grating lobes;
the library does not hide them.
