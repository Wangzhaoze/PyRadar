# Velocity-aware clustering

`rsp.dbscan` implements deterministic DBSCAN for radar points. It accepts either
2D `(x, y)` or 3D `(x, y, z)` geometry and appends scaled radial velocity to the
distance feature:

$$
\mathbf{q}_i=
\begin{bmatrix}
\mathbf{p}_i & s_v v_{r,i}
\end{bmatrix}^{\mathsf T}.
$$

Two points are neighbours when
$\lVert\mathbf{q}_i-\mathbf{q}_j\rVert_2\leq\varepsilon$. `velocityScale` $s_v$
converts metres per second into the spatial distance used by `eps`.

```python
clusters = rsp.dbscan(
    pointCloud,
    eps=0.8,
    minSamples=3,
    velocityScale=0.5,
    dimensions=3,
)
```

The algorithm visits points in input order and uses stable neighbourhood order,
so identical input yields identical labels. Noise points receive label `-1`.

For every cluster, `ClusterSet` reports:

- centroid in FLU coordinates;
- axis-aligned size;
- spatial covariance;
- mean radial velocity and total/aggregate power;
- per-point labels and stable cluster ids.

DBSCAN parameters are sensor- and range-dependent. Angular resolution creates
larger lateral spacing at long range, so one global `eps` may be insufficient for
a very wide operating envelope. Range-adaptive clustering is a future extension,
not an implicit behavior in v1.
