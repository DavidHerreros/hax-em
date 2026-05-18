"""
Point Transformer decoder for CryoEM structural heterogeneity analysis.

This implementation includes three optional architectural changes designed
to suppress per-Gaussian noise at high N (5000-10000 Gaussians):

  1. ``predict_at_coarse_level``:  the head outputs deltas at the
     n2=1024 hierarchy level, which are then interpolated to N points
     using fixed Gaussian-weighted interpolation.  Per-Gaussian deltas
     are no longer independent parameters of the model, so per-particle
     noise can't be absorbed into them.

  2. ``head_rank``:  factorize the head's output projection through a
     low-rank bottleneck.  The output deltas live in a r-dimensional
     basis shared across all points, encoding the prior that real
     protein motions are low-rank (a small number of "modes").

  3. ``use_local_frames``:  the head outputs deltas in a per-Gaussian
     local coordinate frame (PCA of each Gaussian's neighborhood),
     which is then rotated back to global coordinates.  Encodes the
     prior that motions tend to be along structural axes (helices,
     sheets) rather than in arbitrary global directions.

Each flag is independent; you can ablate them by toggling individually.

Output API
----------
``out_channels`` may be an ``int`` (single output array of that channel
count) or a tuple/list of ints (one independent head per entry, returned
as an ``nnx.List``).

When ``use_local_frames=True``, the FIRST head must produce exactly 3
output channels (the position delta), since local-frame rotation only
makes sense for spatial outputs.  Amplitude/sigma heads are unaffected.
"""

from __future__ import annotations
from typing import Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx, struct


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

default_kernel_init = nnx.initializers.lecun_normal()


class MLP(nnx.Module):
    """Two-layer MLP, configurable hidden size, ReLU between, no norm."""

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


# ---------------------------------------------------------------------------
# Point Transformer layer
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Transition Up — feature upsampler used by the trunk
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

@struct.dataclass
class Geometry:
    """All pre-computed geometry tables.

    Beyond the standard fields we add two optional ones used by the new
    architectural toggles:

    * ``local_frames``:  list-wrapped (N, 3, 3) per-Gaussian rotation
      matrix array, taking "local frame" deltas to "global frame"
      deltas.  Used by ``use_local_frames=True``.  Stored as a list of
      one tensor (rather than a bare tensor) so that Orbax can restore
      it in-place via list-element mutation — the dataclass itself is
      immutable.  If unused, contains a single (1, 1, 1) dummy.
    * ``coarse_to_fine_idx`` /
      ``coarse_to_fine_w``: indices and weights for the n2 -> N output
      interpolation.  Used by ``predict_at_coarse_level=True``.
      Identical to ``up_idx[-1]`` / ``up_w[-1]`` (just renamed for
      clarity at the call site).
    """
    positions:  list
    neighbours: list
    rel_pos:    list
    up_idx:     list
    up_w:       list
    res_idx:    list
    res_w:      list
    # New optional field.  Wrapped in a list of length 1 so that the
    # restore path can mutate `geom.local_frames[0] = ...` rather than
    # trying to replace the immutable dataclass attribute.
    local_frames: list


# ---------------------------------------------------------------------------
# Low-rank head: Linear -> small r-dim basis projection -> shared basis
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class PointTransformerDecoder(nnx.Module):
    """Point Transformer decoder of GMM parameters.

    See the module docstring for an overview of the three architectural
    flags this class adds:

      * ``predict_at_coarse_level``  (default True)
      * ``head_rank``                (default None = full rank)
      * ``use_local_frames``         (default False)

    With all three off the model is equivalent to the previous
    reference-faithful implementation.
    """

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
        # --- new architectural toggles ---
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

        # Input MLP:  z -> 128 -> relu -> dropout -> n0*feat -> reshape
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

        # tu3 (n2 -> N) is only used in the OLD (per-point) prediction mode.
        # In the new coarse-level mode we apply the head at n2 and use
        # parameter-free interpolation to N for the output deltas.
        if not predict_at_coarse_level:
            self.tu3 = TransitionUp(feat_dim, feat_dim, rngs=rngs)

        # Residual upsamplers (n0, n1) -> {N or n2 depending on mode}
        # The output of the trunk is concatenated at the level the head
        # operates on, so the residual upsamplers must end at that level.
        self.res_up1 = TransitionUp(feat_dim, feat_dim, rngs=rngs)
        self.res_up2 = TransitionUp(feat_dim, feat_dim, rngs=rngs)

        # ---- head construction ----
        # In the coarse-level mode the head operates on n2 points; in the
        # per-point mode it operates on N.  The head architecture is the
        # same in either case, only its input *spatial* size differs.
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
            # Path A (NEW): apply head at n2, then interpolate deltas to N.
            # Residuals upsampled from (n0, n1) -> n2 (via res_idx tables).
            x1_up = self.res_up1(y0k_1, geom.res_idx[0], geom.res_w[0])
            x2_up = self.res_up2(y0k_2, geom.res_idx[1], geom.res_w[1])

            cat = jnp.concatenate([y0_n2, x1_up, x2_up], axis=-1)
            cat = cat.astype(jnp.float32)                    # (B, n2, 3*feat)

            # Apply head AT THE COARSE LEVEL
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
            # Path B (LEGACY): residuals at n2, then everything to N.
            x_main = self.tu3(y0_n2, geom.up_idx[2], geom.up_w[2])
            x1_at_n2 = self.res_up1(y0k_1, geom.res_idx[0], geom.res_w[0])
            x2_at_n2 = self.res_up2(y0k_2, geom.res_idx[1], geom.res_w[1])
            # Bring residuals from n2 to N using parameter-free interp.
            x1_up = interp_no_proj(x1_at_n2, geom.up_idx[2], geom.up_w[2])
            x2_up = interp_no_proj(x2_at_n2, geom.up_idx[2], geom.up_w[2])

            cat = jnp.concatenate([x_main, x1_up, x2_up], axis=-1)
            cat = cat.astype(jnp.float32)                    # (B, N, 3*feat)

            if self._multi_head:
                outs = [self.output_scale * h(cat) for h in self.head]
            else:
                outs = self.output_scale * self.head(cat)

        # ---- local-frame rotation (applied to position deltas only) ----
        if self.use_local_frames:
            # geom.local_frames is a list of one (N, 3, 3) tensor
            # (list-wrapped for serialization-friendly mutation).
            frames = geom.local_frames[0]                    # (N, 3, 3)
            if isinstance(outs, list):
                # First head is positions -> (B, N, 3)
                pos = outs[0]
                # Rotate: out_global[b, n, :] = frames[n] @ pos[b, n, :]
                pos_rotated = jnp.einsum("nij,bnj->bni", frames, pos)
                outs = [pos_rotated] + outs[1:]
            else:
                # Single head, must be (B, N, 3)
                outs = jnp.einsum("nij,bnj->bni", frames, outs)

        return outs


# ---------------------------------------------------------------------------
# Geometry pre-computation
# ---------------------------------------------------------------------------

def _compute_local_frames(positions: np.ndarray,
                          neighbours: np.ndarray) -> np.ndarray:
    """For each point, compute a 3x3 rotation matrix via PCA of its
    k-nearest neighbours.

    The columns of the returned matrix are the local axes (largest-,
    medium-, smallest-variance directions).  Multiplying a "local"
    vector by this matrix yields its "global" expression.

    Returns shape (N, 3, 3).  At init, applying these frames to small
    random deltas produces small random global deltas with the same
    statistics, so this is a structure-preserving change of coordinates.
    """
    N = positions.shape[0]
    K = neighbours.shape[1]
    frames = np.zeros((N, 3, 3), dtype=np.float32)
    for i in range(N):
        nbrs = positions[neighbours[i]]                  # (K, 3)
        centered = nbrs - nbrs.mean(axis=0, keepdims=True)
        # SVD-based PCA — numerically stable and works for K >= 3.
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        # vh rows are the principal directions, ordered by descending variance.
        # Build R so that columns are local axes; then global = R @ local.
        R = vh.T                                          # (3, 3)
        # Ensure right-handed (det > 0)
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

    # ---- hierarchical point sets via k-means, snapped to actual GMM
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

    # ---- k-NN inside each PT layer (for self-attention)
    neighbours, rel_pos = [], []
    for p in coarse_positions:
        nn_ = NearestNeighbors(n_neighbors=k_attn).fit(p)
        idx = nn_.kneighbors(p, return_distance=False).astype(np.int32)
        rel = p[:, None, :] - p[idx]    # centre - neighbour, matches reference
        neighbours.append(jnp.asarray(idx))
        rel_pos.append(jnp.asarray(rel))

    # ---- auto gauss_scale
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

    # ---- sequential upsampling P_i -> P_{i+1}
    up_idx, up_w = [], []
    for i in range(len(all_positions) - 1):
        coarse, fine = all_positions[i], all_positions[i + 1]
        nn_ = NearestNeighbors(n_neighbors=k_up).fit(coarse)
        d, idx = nn_.kneighbors(fine, return_distance=True)
        up_idx.append(jnp.asarray(idx.astype(np.int32)))
        up_w.append(jnp.asarray(_gaussian_weights(d)))

    # ---- residual upsampling P_0 -> N and P_1 -> N.
    # NOTE: when predict_at_coarse_level=True the residuals are
    # consumed at the *n2* level rather than at N, so we want
    # P_0 -> n2 and P_1 -> n2.  But we don't know which mode the user
    # will pick at geometry-build time.  Solution: store BOTH paths
    # (to n2 and to N) — but that doubles the geometry size for a small
    # gain.  Simpler: store paths to N, and use the existing up_idx[1]
    # / up_idx[2] tables to route through n2 when in coarse-level mode.
    #
    # Decision: we store paths *to n2*, since that's the more
    # information-conservative target (interpolating to n2 then
    # parameter-free expanding to N preserves locality better than
    # interpolating directly to N).  For legacy mode we route via
    # the existing tu3 path anyway, so this only affects how the
    # *residuals* end up.
    res_idx, res_w = [], []
    fine_for_res = all_positions[2]                     # n2-level points
    for i in (0, 1):
        coarse = all_positions[i]
        nn_ = NearestNeighbors(n_neighbors=k_up).fit(coarse)
        d, idx = nn_.kneighbors(fine_for_res, return_distance=True)
        res_idx.append(jnp.asarray(idx.astype(np.int32)))
        res_w.append(jnp.asarray(_gaussian_weights(d)))

    # ---- local frames (optional)
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