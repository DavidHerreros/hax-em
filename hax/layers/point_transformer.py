from __future__ import annotations
from typing import Sequence, Union
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx, struct


default_kernel_init = nnx.initializers.lecun_normal()


class MLP(nnx.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        last_kernel_init: nnx.Initializer = default_kernel_init,
        last_dtype: jnp.dtype = jnp.float32,
        last_use_bias: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        self.fc1 = nnx.Linear(in_dim, hidden_dim, dtype=jnp.bfloat16, rngs=rngs)
        self.fc2 = nnx.Linear(
            hidden_dim, out_dim,
            kernel_init=last_kernel_init,
            use_bias=last_use_bias,
            dtype=last_dtype,
            rngs=rngs,
        )

    def __call__(self, x):
        return self.fc2(nnx.relu(self.fc1(x)))


class PointTransformerLayer(nnx.Module):
    def __init__(self, dim: int, nk: int = 512, *, rngs: nnx.Rngs,
                 dropout_rate: float = 0.2):
        self.dim = dim
        self.nk  = nk
        self.phi = nnx.Linear(dim, nk, dtype=jnp.bfloat16, rngs=rngs)
        self.psi = nnx.Linear(dim, nk, dtype=jnp.bfloat16, rngs=rngs)
        self.theta_in  = nnx.Linear(3, 3,  dtype=jnp.bfloat16, rngs=rngs)
        self.theta_out = nnx.Linear(3, nk, dtype=jnp.bfloat16, rngs=rngs)
        self.gamma1 = nnx.Linear(nk, nk, dtype=jnp.bfloat16, rngs=rngs)
        self.gamma2 = nnx.Linear(nk, nk, dtype=jnp.bfloat16, rngs=rngs)
        self.value_proj = nnx.Linear(nk, nk, dtype=jnp.bfloat16, rngs=rngs)
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)

    def __call__(self, x, neighbour_idx, rel_pos, *, deterministic=False):
        q = self.phi(x)
        k = self.psi(x)
        k_nb = k[:, neighbour_idx, :]
        delta = nnx.relu(self.theta_in(rel_pos))
        delta = self.theta_out(delta)[None, ...]
        qk_delta = q[:, :, None, :] - k_nb + delta
        w = nnx.relu(qk_delta)
        w = self.gamma1(w)
        w = nnx.relu(w)
        w = self.gamma2(w)
        w = jax.nn.softmax(w, axis=2)
        v = self.value_proj(k_nb) + delta
        out = jnp.sum(w * v, axis=2)
        return self.dropout(out, deterministic=deterministic)


class PTBlock(nnx.Module):
    def __init__(self, dim: int, nk: int = 512, *, rngs: nnx.Rngs,
                 attn_dropout: float = 0.2, mid_dropout: float = 0.2):
        self.dim = dim
        self.nk  = nk
        self.attn = PointTransformerLayer(dim, nk=nk, rngs=rngs,
                                          dropout_rate=attn_dropout)
        self.lin_mid = nnx.Linear(dim + nk, nk, dtype=jnp.bfloat16, rngs=rngs)
        self.mid_dropout = nnx.Dropout(mid_dropout, rngs=rngs)
        self.lin_out = nnx.Linear(nk, dim, dtype=jnp.bfloat16, rngs=rngs)

    def __call__(self, y0, neighbour_idx, rel_pos, *, deterministic=False):
        y1 = self.attn(y0, neighbour_idx, rel_pos,
                       deterministic=deterministic)
        y1 = jnp.concatenate([y0, y1], axis=-1)
        yk = self.lin_mid(y1)
        yk = self.mid_dropout(yk, deterministic=deterministic)
        yk = self.lin_out(yk)
        return y0 + yk


class TransitionUp(nnx.Module):
    def __init__(self, in_dim: int, out_dim: int, *, rngs: nnx.Rngs):
        self.proj = nnx.Linear(in_dim, out_dim, dtype=jnp.bfloat16, rngs=rngs)

    def __call__(self, x_coarse, upsample_idx, upsample_w):
        x = self.proj(x_coarse)
        gathered = x[:, upsample_idx, :]
        return jnp.sum(gathered * upsample_w[None, :, :, None], axis=2)


def interp_no_proj(x_coarse, upsample_idx, upsample_w):
    """Pure Gaussian-weighted interpolation, no learned projection.

    Used in change #1 to interpolate the *output deltas* (not features)
    from n2 to N points.  Because there's no learnable layer, the only
    way to get per-Gaussian deltas is via interpolation of the n2-point
    output — which guarantees local smoothness.
    """
    gathered = x_coarse[:, upsample_idx, :]
    return jnp.sum(gathered * upsample_w[None, :, :, None], axis=2)


@struct.dataclass
class Geometry:
    positions:  list
    neighbours: list
    rel_pos:    list
    up_idx:     list
    up_w:       list
    res_idx:    list
    res_w:      list
    local_frames: list

class LowRankHead(nnx.Module):
    """Output head with explicit rank-r factorization.

    output = (input @ W_in)  @  W_basis
        W_in    : (in_dim, r)
        W_basis : (r, out_dim)

    The basis matrix W_basis is shared across all points and particles.
    Real deltas are thus expressed as linear combinations of ``r``
    learned "modes".
    """

    def __init__(self, in_dim: int, out_dim: int, rank: int, *,
                 init_std: float = 1e-4, rngs: nnx.Rngs):
        # per-point projection to r-dim coefficients
        self.coef = nnx.Linear(
            in_dim, rank,
            kernel_init=nnx.initializers.normal(stddev=init_std),
            use_bias=False, dtype=jnp.float32, rngs=rngs,
        )
        # shared basis matrix (r, out_dim).  Initialized with
        # lecun-normal so each mode is a meaningful "direction" in
        # output space at init.
        self.basis = nnx.Param(
            nnx.initializers.lecun_normal()(rngs.params(), (rank, out_dim))
        )

    def __call__(self, x):
        # x: (B, N, in_dim) -> (B, N, rank) -> (B, N, out_dim)
        c = self.coef(x)
        return c @ self.basis.get_value()


class PointTransformerDecoder(nnx.Module):
    def __init__(
        self,
        latent_dim: int,
        feat_dim: int = 256,
        nk: int = 512,
        hierarchical_sizes: Sequence[int] = (64, 256, 1024),
        input_bottleneck: int = 128,
        out_channels: Union[int, Sequence[int]] = 5,
        output_scale: float = 0.5,
        final_init_std: float = 1e-4,
        input_dropout: float = 0.1,
        attn_dropout: float = 0.2,
        mid_dropout: float = 0.2,
        predict_at_coarse_level: bool = True,
        head_rank: int | None = None,
        use_local_frames: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        assert len(hierarchical_sizes) == 3, \
            "Paper architecture has exactly 3 PT layers"
        self.feat_dim = feat_dim
        self.nk = nk
        self.n0, self.n1, self.n2 = hierarchical_sizes
        self.output_scale = output_scale
        self.predict_at_coarse_level = predict_at_coarse_level
        self.head_rank = head_rank
        self.use_local_frames = use_local_frames

        # If using local frames, validate that the first head outputs
        # exactly 3 channels (position delta).
        if use_local_frames:
            if isinstance(out_channels, int):
                assert out_channels == 3, (
                    "use_local_frames=True requires the head's first 3 "
                    "channels to be position deltas; pass out_channels=3 "
                    "or out_channels=(3, ...)."
                )
            else:
                assert out_channels[0] == 3, (
                    "use_local_frames=True requires the first head to "
                    "output exactly 3 channels (position delta)."
                )

        # Input MLP:  z -> 128 -> relu -> dropout -> n0 * feat -> reshape
        self.in_lin1 = nnx.Linear(latent_dim, input_bottleneck,
                                  dtype=jnp.bfloat16, rngs=rngs)
        self.in_dropout = nnx.Dropout(input_dropout, rngs=rngs)
        self.in_lin2 = nnx.Linear(input_bottleneck, self.n0 * feat_dim,
                                  dtype=jnp.bfloat16, rngs=rngs)

        # 3 PT blocks
        self.pt1 = PTBlock(feat_dim, nk=nk, rngs=rngs,
                           attn_dropout=attn_dropout, mid_dropout=mid_dropout)
        self.pt2 = PTBlock(feat_dim, nk=nk, rngs=rngs,
                           attn_dropout=attn_dropout, mid_dropout=mid_dropout)
        self.pt3 = PTBlock(feat_dim, nk=nk, rngs=rngs,
                           attn_dropout=attn_dropout, mid_dropout=mid_dropout)

        # Sequential upsamplers  n0 -> n1 -> n2
        self.tu1 = TransitionUp(feat_dim, feat_dim, rngs=rngs)
        self.tu2 = TransitionUp(feat_dim, feat_dim, rngs=rngs)

        if not predict_at_coarse_level:
            self.tu3 = TransitionUp(feat_dim, feat_dim, rngs=rngs)

        # Residual upsamplers (n0, n1) -> {N or n2 depending on mode}
        self.res_up1 = TransitionUp(feat_dim, feat_dim, rngs=rngs)
        self.res_up2 = TransitionUp(feat_dim, feat_dim, rngs=rngs)

        # Head construction
        def _make_head(c):
            if head_rank is not None:
                return LowRankHead(
                    3 * feat_dim, c, rank=head_rank,
                    init_std=final_init_std, rngs=rngs,
                )
            if final_init_std == 0.0:
                kernel_init = nnx.initializers.zeros_init()
            else:
                kernel_init = nnx.initializers.normal(stddev=final_init_std)
            return nnx.Linear(
                3 * feat_dim, c,
                kernel_init=kernel_init,
                use_bias=False, dtype=jnp.float32, rngs=rngs,
            )

        if isinstance(out_channels, int):
            self.head = _make_head(out_channels)
            self._multi_head = False
        elif isinstance(out_channels, (list, tuple)):
            self.head = nnx.List([_make_head(c) for c in out_channels])
            self._multi_head = True
        else:
            raise TypeError(
                "out_channels must be int, list[int], or tuple[int]; "
                f"got {type(out_channels).__name__}"
            )

    def __call__(
        self,
        z: jax.Array,
        geom: Geometry,
        *,
        deterministic: bool = False,
    ) -> Union[jax.Array, Sequence[jax.Array]]:
        B = z.shape[0]

        # Input MLP
        x = self.in_lin1(z)
        x = nnx.relu(x)
        x = self.in_dropout(x, deterministic=deterministic)
        x = self.in_lin2(x)
        x = x.reshape(B, self.n0, self.feat_dim)

        # PT block 1 + residual store
        y0 = self.pt1(x, geom.neighbours[0], geom.rel_pos[0],
                      deterministic=deterministic)
        y0k_1 = y0
        y0 = self.tu1(y0, geom.up_idx[0], geom.up_w[0])

        # PT block 2 + residual store
        y0 = self.pt2(y0, geom.neighbours[1], geom.rel_pos[1],
                      deterministic=deterministic)
        y0k_2 = y0
        y0 = self.tu2(y0, geom.up_idx[1], geom.up_w[1])

        # PT block 3 (operates on n2 points)
        y0_n2 = self.pt3(y0, geom.neighbours[2], geom.rel_pos[2],
                         deterministic=deterministic)        # (B, n2, feat)

        if self.predict_at_coarse_level:
            x1_up = self.res_up1(y0k_1, geom.res_idx[0], geom.res_w[0])
            x2_up = self.res_up2(y0k_2, geom.res_idx[1], geom.res_w[1])

            cat = jnp.concatenate([y0_n2, x1_up, x2_up], axis=-1)
            cat = cat.astype(jnp.float32)                    # (B, n2, 3*feat)

            # Apply head (coarse level)
            if self._multi_head:
                outs_coarse = [h(cat) for h in self.head]
            else:
                outs_coarse = self.head(cat)

            # Interpolate from n2 -> N using parameter-free Gaussian
            # weights (the existing up_idx[2] / up_w[2] tables).
            if isinstance(outs_coarse, list):
                outs = [
                    self.output_scale *
                    interp_no_proj(o, geom.up_idx[2], geom.up_w[2])
                    for o in outs_coarse
                ]
            else:
                outs = self.output_scale * interp_no_proj(
                    outs_coarse, geom.up_idx[2], geom.up_w[2]
                )
        else:
            x_main = self.tu3(y0_n2, geom.up_idx[2], geom.up_w[2])
            x1_at_n2 = self.res_up1(y0k_1, geom.res_idx[0], geom.res_w[0])
            x2_at_n2 = self.res_up2(y0k_2, geom.res_idx[1], geom.res_w[1])
            x1_up = interp_no_proj(x1_at_n2, geom.up_idx[2], geom.up_w[2])
            x2_up = interp_no_proj(x2_at_n2, geom.up_idx[2], geom.up_w[2])

            cat = jnp.concatenate([x_main, x1_up, x2_up], axis=-1)
            cat = cat.astype(jnp.float32)                    # (B, N, 3 * feat)

            if self._multi_head:
                outs = [self.output_scale * h(cat) for h in self.head]
            else:
                outs = self.output_scale * self.head(cat)

        # Local-frame rotation
        if self.use_local_frames:
            frames = geom.local_frames[0]                    # (N, 3, 3)
            if isinstance(outs, list):
                pos = outs[0]
                pos_rotated = jnp.einsum("nij,bnj->bni", frames, pos)
                outs = [pos_rotated] + outs[1:]
            else:
                outs = jnp.einsum("nij,bnj->bni", frames, outs)

        return outs


def _compute_local_frames(positions: np.ndarray,
                          neighbours: np.ndarray) -> np.ndarray:
    N = positions.shape[0]
    frames = np.zeros((N, 3, 3), dtype=np.float32)
    for i in range(N):
        nbrs = positions[neighbours[i]]                  # (K, 3)
        centered = nbrs - nbrs.mean(axis=0, keepdims=True)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        R = vh.T                                          # (3, 3)
        if np.linalg.det(R) < 0:
            R[:, -1] *= -1
        frames[i] = R
    return frames


def build_geometry(
    gmm_positions: np.ndarray,
    hierarchical_sizes: Sequence[int] = (64, 256, 1024),
    k_attn: int = 16,
    k_up: int | None = None,
    gauss_scale: float | str = "auto",
    compute_local_frames: bool = False,
    k_frame: int = 16,
    *,
    seed: int = 0,
) -> Geometry:
    """Build the geometry tables.

    Parameters
    ----------
    gmm_positions
        (N, 3) float array — centres of the consensus GMM.
    hierarchical_sizes
        Number of points at each PT layer, listed coarse-to-fine.
    k_attn, k_up
        Self-attention and upsample neighbour counts.
    gauss_scale
        Float, or ``"auto"`` (default) to set automatically based on
        median neighbour distance.
    compute_local_frames
        If True, compute per-Gaussian local frames (3x3 rotation matrices
        from PCA of the k_frame nearest neighbours).  Required for the
        ``use_local_frames=True`` decoder option.
    k_frame
        Number of neighbours used for local-frame PCA.  Default 16.
    """
    from sklearn.cluster import KMeans
    from sklearn.neighbors import NearestNeighbors
    import scipy.spatial.distance as scipydist

    if k_up is None:
        k_up = max(1, k_attn // 2)

    gmm_positions = np.asarray(gmm_positions, dtype=np.float32)
    N = gmm_positions.shape[0]
    assert hierarchical_sizes[-1] < N, \
        "the last PT layer must be coarser than the final GMM"

    # Hierarchical point via k-means
    coarse_positions: list[np.ndarray] = []
    for n in hierarchical_sizes:
        km = KMeans(n_clusters=n, n_init=10, max_iter=100,
                    random_state=seed).fit(gmm_positions)
        centres = km.cluster_centers_.astype(np.float32)
        d = scipydist.cdist(centres, gmm_positions)
        nearest_idx = np.argmin(d, axis=1)
        snapped = gmm_positions[nearest_idx]
        coarse_positions.append(snapped)
    all_positions = coarse_positions + [gmm_positions]

    # k-NN inside each PT layer (for self-attention)
    neighbours, rel_pos = [], []
    for p in coarse_positions:
        nn_ = NearestNeighbors(n_neighbors=k_attn).fit(p)
        idx = nn_.kneighbors(p, return_distance=False).astype(np.int32)
        rel = p[:, None, :] - p[idx]    # centre - neighbour, matches reference
        neighbours.append(jnp.asarray(idx))
        rel_pos.append(jnp.asarray(rel))

    # auto gauss_scale
    if isinstance(gauss_scale, str):
        assert gauss_scale == "auto"
        coarse_probe, fine_probe = all_positions[-2], all_positions[-1]
        nn_probe = NearestNeighbors(n_neighbors=k_up).fit(coarse_probe)
        d_probe, _ = nn_probe.kneighbors(fine_probe, return_distance=True)
        d_median = float(np.median(d_probe))
        gauss_scale = 2.0 / (d_median ** 2) if d_median > 0 else 50.0
        print(f"  [build_geometry] gauss_scale auto-set to {gauss_scale:.3f} "
              f"(d_median = {d_median:.4f})")

    def _gaussian_weights(d):
        logits = -(d ** 2) * gauss_scale
        logits = logits - logits.max(axis=1, keepdims=True)
        w = np.exp(logits).astype(np.float32)
        return w / w.sum(axis=1, keepdims=True)

    # Sequential upsampling P_i -> P_{i+1}
    up_idx, up_w = [], []
    for i in range(len(all_positions) - 1):
        coarse, fine = all_positions[i], all_positions[i + 1]
        nn_ = NearestNeighbors(n_neighbors=k_up).fit(coarse)
        d, idx = nn_.kneighbors(fine, return_distance=True)
        up_idx.append(jnp.asarray(idx.astype(np.int32)))
        up_w.append(jnp.asarray(_gaussian_weights(d)))

    # Residual upsampling P_0 -> N and P_1 -> N.
    res_idx, res_w = [], []
    fine_for_res = all_positions[2]                     # n2-level points
    for i in (0, 1):
        coarse = all_positions[i]
        nn_ = NearestNeighbors(n_neighbors=k_up).fit(coarse)
        d, idx = nn_.kneighbors(fine_for_res, return_distance=True)
        res_idx.append(jnp.asarray(idx.astype(np.int32)))
        res_w.append(jnp.asarray(_gaussian_weights(d)))

    # Local frames (optional)
    if compute_local_frames:
        nn_frame = NearestNeighbors(n_neighbors=k_frame).fit(gmm_positions)
        frame_idx = nn_frame.kneighbors(gmm_positions, return_distance=False)
        frames = _compute_local_frames(gmm_positions, frame_idx)
        local_frames = [jnp.asarray(frames)]            # list of one (N, 3, 3)
    else:
        local_frames = [jnp.zeros((1, 1, 1), dtype=jnp.float32)]

    return Geometry(
        positions=list(jnp.asarray(p) for p in all_positions),
        neighbours=list(neighbours),
        rel_pos=list(rel_pos),
        up_idx=list(up_idx),
        up_w=list(up_w),
        res_idx=list(res_idx),
        res_w=list(res_w),
        local_frames=local_frames,
    )


@partial(jax.jit, static_argnums=(1,))
def farthest_point_sample(points, n_samples, start_idx=0):
    """Farthest-point sampling -> indices of a near-uniform subset of the support.

    points:    (N, D)
    n_samples: int (<= N)  -- fixed output size, good for batching net2
    returns:   (n_samples,) integer indices into `points`

    Idea: greedily pick the point farthest from everything chosen so far, so the
    selected set spreads evenly over the *shape* regardless of the original
    sampling density. Fixed output size -> erases density variation at the door.
    """
    N = points.shape[0]

    def update_min_dist(idx, dists):
        d = jnp.sum((points - points[idx]) ** 2, axis=-1)  # (N,)
        return jnp.minimum(dists, d)

    dists0 = update_min_dist(start_idx, jnp.full((N,), jnp.inf))

    def step(dists, _):
        next_idx = jnp.argmax(dists)
        dists = update_min_dist(next_idx, dists)
        return dists, next_idx

    _, rest = jax.lax.scan(step, dists0, None, length=n_samples - 1)
    return jnp.concatenate([jnp.array([start_idx], dtype=rest.dtype), rest])


def fps_resample(points, n_samples, start_idx=0):
    """Convenience wrapper returning the resampled coordinates (n_samples, D)."""
    idx = farthest_point_sample(points, n_samples, start_idx)
    return points[idx]


# Batched version (same n_samples for every cloud in the batch).
fps_resample_batched = jax.jit(
    jax.vmap(fps_resample, in_axes=(0, None, None)),
    static_argnums=(1,),
)


class PointMLPBlock(nnx.Module):
    """Residual per-point MLP block. Pointwise -> permutation-equivariant."""

    def __init__(
            self,
            dim: int,
            expansion: int = 2,
            *,
            dtype: jnp.dtype = jnp.float32,
            param_dtype: jnp.dtype = jnp.float32,
            rngs: nnx.Rngs,
    ):
        # LayerNorm: keep computation in fp32 for stability, params in fp32.
        self.norm = nnx.LayerNorm(
            dim, dtype=jnp.float32, param_dtype=jnp.float32, rngs=rngs
        )
        self.fc1 = nnx.Linear(
            dim, dim * expansion, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )
        self.fc2 = nnx.Linear(
            dim * expansion, dim, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )
        self.dtype = dtype

    def __call__(self, x):  # x: (B, N, dim) in self.dtype
        # LayerNorm in fp32, then cast back to compute dtype.
        h = self.norm(x.astype(jnp.float32)).astype(self.dtype)
        h = nnx.gelu(self.fc1(h))
        h = self.fc2(h)
        return x + h


class PointCloudEncoder(nnx.Module):
    """
    (B, N, 3) point clouds -> (B, latent_dim) latent vectors.
    Permutation-invariant. Variable N supported via optional (B, N) bool mask.
    """

    def __init__(
            self,
            latent_dim: int,
            hidden: int = 256,
            n_blocks: int = 4,
            scale_multiplier: float = 1.0,
            *,
            dtype: jnp.dtype = jnp.bfloat16,
            rngs: nnx.Rngs,
    ):
        self.scale_multiplier = scale_multiplier
        self.dtype = dtype
        self.embed = nnx.Linear(
            3, hidden, dtype=dtype, rngs=rngs
        )
        self.blocks = nnx.List([
            PointMLPBlock(
                hidden, dtype=dtype, rngs=rngs
            )
            for _ in range(n_blocks)
        ])
        self.point_norm = nnx.LayerNorm(
            hidden, dtype=jnp.float32, rngs=rngs
        )

        head_in = 2 * hidden  # [max || mean]
        self.head_norm = nnx.LayerNorm(
            head_in, dtype=jnp.float32, rngs=rngs
        )
        self.head1 = nnx.Linear(
            head_in, hidden, dtype=dtype, rngs=rngs
        )
        self.head2 = nnx.Linear(
            hidden, latent_dim, dtype=dtype, rngs=rngs
        )

    def __call__(self, cloud, mask=None, ):
        # Cast input to compute dtype.
        x = cloud.astype(self.dtype)  # (B, N, 3)
        h = self.embed(x)  # (B, N, H)
        for block in self.blocks:
            h = block(h)
        h = self.point_norm(h.astype(jnp.float32)).astype(self.dtype)

        if mask is None:
            pooled_max = jnp.max(h, axis=1)
            pooled_mean = jnp.mean(h, axis=1)
        else:
            m = mask[..., None]  # (B, N, 1)
            mf = m.astype(h.dtype)
            neg_inf = jnp.array(-jnp.inf, dtype=h.dtype)
            pooled_max = jnp.max(jnp.where(m, h, neg_inf), axis=1)
            denom = jnp.clip(jnp.sum(mf, axis=1), min=jnp.array(1.0, dtype=h.dtype))
            pooled_mean = jnp.sum(h * mf, axis=1) / denom

        g = jnp.concatenate([pooled_max, pooled_mean], axis=-1)  # (B, 2H)
        g = self.head_norm(g.astype(jnp.float32)).astype(self.dtype)
        g = nnx.gelu(self.head1(g))
        return self.scale_multiplier * self.head2(g)  # (B, latent_dim) in self.dtype
