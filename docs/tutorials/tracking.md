# Constant-velocity multi-target tracking

The tracker supports 2D or 3D FLU state with Cartesian position and velocity:

$$
\mathbf{x}=
\begin{bmatrix}
\mathbf{p}^{\mathsf T} & \mathbf{v}^{\mathsf T}
\end{bmatrix}^{\mathsf T}.
$$

## Prediction

For frame interval $\Delta t$,

$$
\mathbf{F}=
\begin{bmatrix}
\mathbf{I} & \Delta t\mathbf{I}\\
\mathbf{0} & \mathbf{I}
\end{bmatrix}.
$$

White acceleration noise is mapped with

$$
\mathbf{G}=
\begin{bmatrix}
\frac{1}{2}\Delta t^2\mathbf{I}\\
\Delta t\mathbf{I}
\end{bmatrix},
\qquad
\mathbf{Q}=\sigma_a^2\mathbf{G}\mathbf{G}^{\mathsf T}.
$$

Timestamps produce variable $\Delta t$. Without timestamps, callers must supply a
positive interval or the pipeline uses the model frame period.

## Measurements

Cartesian point or cluster centroids use a linear Kalman update. Radial velocity
uses the nonlinear observation

$$
h(\mathbf{x})=\frac{\mathbf{p}^{\mathsf T}\mathbf{v}}
{\lVert\mathbf{p}\rVert_2}
$$

and an EKF Jacobian. Near the origin, radial velocity is ignored because the
direction is undefined.

## Association

Each track/measurement pair receives squared Mahalanobis distance

$$
d^2=\mathbf{y}^{\mathsf T}\mathbf{S}^{-1}\mathbf{y}.
$$

Pairs outside `gatingThreshold` are forbidden. Hungarian assignment solves the
remaining global nearest-neighbour problem. Unmatched measurements start tracks;
unmatched tracks coast.

## Lifecycle

- `tentative`: born but below `confirmationHits`;
- `confirmed`: accumulated enough associated measurements;
- `coasting`: confirmed and currently missed;
- `deleted`: reached `deletionMisses` and appears once in the returned snapshot
  before removal.

```python
tracker = rsp.MultiTargetTracker(
    TrackingConfig(
        enabled=True,
        dimensions=3,
        confirmationHits=3,
        deletionMisses=5,
    )
)
tracks = tracker.update(points, timestamp=frameTimestamp)
```

The tracker is stateful; call `reset()` between independent sequences. Its radial
velocity is a line-of-sight observation, not full Cartesian velocity. Reliable
tangential velocity emerges only from position evolution over time.
