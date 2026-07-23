import jax
import jax.numpy as jnp


def _distance_activation(pairwise_distances, mean_neighbour_distance):
    """A continuous assignment of 'neighbour-like-ness'."""
    cutoff = mean_neighbour_distance
    # Clamp distances between tau and 1.5 * tau
    x2 = jnp.clip(pairwise_distances, a_min=cutoff, a_max=1.5 * cutoff)
    # Quadratic kernel logic from DynaMight
    activation = (1.0 - (4.0 / cutoff ** 2.) * (x2 - cutoff) ** 2.) ** 2.
    return activation


def _neighbour_activation(neighbours_per_point, minimum=1.0, maximum=3.0):
    """Quadratic penalisation on number of neighbours outside range."""
    # Penalty for too few neighbors
    x1 = jnp.clip(neighbours_per_point, a_max=minimum)
    # Penalty for too many neighbors
    x2 = jnp.clip(neighbours_per_point, a_min=maximum)
    return (x1 - minimum) ** 2. + (x2 - maximum) ** 2.


def calculate_deformation_regularity_loss(positions, radius_graph, consensus_distances, edge_weights,
                                          eps=1e-8, weighted_mean=False):
    """Preserves local distances (Spring-like prior)"""
    i, j = radius_graph
    # Safe distance calculation
    diffs = positions[i] - positions[j]
    distances = jnp.sqrt(jnp.sum(diffs ** 2., axis=-1) + eps)

    # Square error compared to consensus
    loss = (distances - consensus_distances) ** 2.
    if weighted_mean:
        return jnp.sum(edge_weights * loss) / (jnp.sum(edge_weights) + eps)
    return jnp.mean(edge_weights * loss)

def _closest_rotation_svd(S):
    """Proper rotation ``R = V diag(1,1,det) U^T`` from the SVD of ``S = U Σ V^T``.

    ``S`` is ``(..., 3, 3)``. Exact and robust (SVD orders the singular values, so
    the reflection fix flips the *smallest* singular direction), but batched 3x3
    SVD is a severe XLA:GPU bottleneck — see :func:`_closest_rotation_polar`.
    """
    U, _, Vt = jnp.linalg.svd(S)
    V = jnp.swapaxes(Vt, -1, -2)
    Ut = jnp.swapaxes(U, -1, -2)
    R = jnp.einsum('...ij,...jk->...ik', V, Ut)                  # V U^T
    det = jnp.linalg.det(R)
    D = jnp.ones(S.shape[:-1], dtype=S.dtype).at[..., -1].set(jnp.sign(det))
    return jnp.einsum('...ij,...j,...jk->...ik', V, D, Ut)       # V diag(1,1,sign) U^T


def _closest_rotation_polar(S, iters=6):
    """Proper rotation closest to ``S`` via a scaled polar iteration (SVD-free).

    The ARAP rotation ``R = V U^T`` (for ``S = U Σ V^T``) is the transpose of the
    orthogonal polar factor of ``S`` (``S = Q P`` with ``Q = U V^T``), so we solve
    for ``Q`` with Higham's determinant-scaled Newton iteration
    ``Q ← ½ (γ Q + γ^{-1} Q^{-T})`` and return ``R = Q^T``. The iteration uses only
    batched 3x3 matmuls and inverses — exactly the ops XLA:GPU fuses efficiently —
    and converges in ~4-6 steps in the near-identity regime ARAP operates in
    (measured agreement with the SVD rotation: ``max|R - R_svd| ≈ 2e-6``), while
    being ~28x faster on GPU for a ``(256, 30000, 3, 3)`` batch.

    Reflection guard: a genuine reflection (``det < 0``) would need the smallest
    singular direction to fix properly, which we do not have without an SVD. Such
    nodes are pathological (folded / degenerate neighbourhoods) and are already
    pushed towards ``S ≈ eps·I`` (hence ``R ≈ I``) by the caller's regularisation,
    so we fall back to the identity there — a conservative, stable choice that
    keeps ``R`` a proper rotation. ``S`` is ``(..., 3, 3)``.
    """
    Q = S
    for _ in range(iters):
        Qinv = jnp.linalg.inv(Q)
        QinvT = jnp.swapaxes(Qinv, -1, -2)
        # 1-norm/inf-norm scaling (Higham) for fast, stable convergence.
        n1 = jnp.max(jnp.sum(jnp.abs(Q), axis=-2), axis=-1)
        ni = jnp.max(jnp.sum(jnp.abs(Q), axis=-1), axis=-1)
        m1 = jnp.max(jnp.sum(jnp.abs(Qinv), axis=-2), axis=-1)
        mi = jnp.max(jnp.sum(jnp.abs(Qinv), axis=-1), axis=-1)
        g = (((m1 * mi) / (n1 * ni)) ** 0.25)[..., None, None]
        Q = 0.5 * (g * Q + (1.0 / g) * QinvT)

    R = jnp.swapaxes(Q, -1, -2)                                  # R = Q^T = V U^T
    det = jnp.linalg.det(R)
    eye = jnp.broadcast_to(jnp.eye(3, dtype=R.dtype), R.shape)
    return jnp.where((det < 0.0)[..., None, None], eye, R)


def calculate_arap_loss(positions, consensus_positions, radius_graph, edge_weights, num_points,
                        eps=1e-6, rotation_method="polar", polar_iters=6, weighted_mean=False):
    """As-rigid-as-possible (ARAP) energy.

    For every node the best-fit local *rotation* is factored out before the
    deformation is penalised, so a locally rigid motion (rotation of a domain)
    costs nothing and only the genuinely non-rigid distortion is penalised. This
    is a stronger, more physical prior than distance preservation alone
    (:func:`calculate_deformation_regularity_loss`), which also penalises rigid
    local rotations and therefore over-stiffens hinge/domain motions while being
    weaker against noise-driven shear.

    ``E = Σ_i Σ_{j∈N(i)} w_ij || (p'_i - p'_j) - R_i (p_i - p_j) ||²`` where the
    optimal ``R_i`` is recovered per node from the local covariance
    ``S_i = Σ_j w_ij (p_i - p_j)(p'_i - p'_j)^T`` (Sorkine & Alexa 2007).

    Parameters
    ----------
    positions
        ``(N, 3)`` deformed point positions (``p'``).
    consensus_positions
        ``(N, 3)`` rest/consensus positions (``p``).
    radius_graph
        ``(2, E)`` edge index ``(i, j)`` (same graph as the other graph losses).
    edge_weights
        ``(E,)`` per-edge weights.
    num_points
        ``N`` (static).
    rotation_method
        How to recover the per-node rotation ``R_i``. ``"polar"`` (default) uses a
        matmul-only scaled polar iteration (:func:`_closest_rotation_polar`) — far
        faster than SVD on GPU and accurate to ~1e-6 here. ``"svd"`` uses the exact
        SVD path (:func:`_closest_rotation_svd`); keep it as a numerical reference /
        fallback (run both and compare the loss to self-check the polar path).
    polar_iters
        Number of polar iterations when ``rotation_method="polar"`` (~4-6 suffice).
    """
    i, j = radius_graph
    e0 = consensus_positions[i] - consensus_positions[j]        # rest edge   (E, 3)
    e = positions[i] - positions[j]                             # deformed    (E, 3)
    w = edge_weights

    # Per-node covariance S_i = Σ_j w_ij e0 e^T, accumulated at node i.
    outer = w[:, None, None] * (e0[:, :, None] * e[:, None, :])  # (E, 3, 3)
    S = jax.ops.segment_sum(outer, i, num_segments=num_points)   # (N, 3, 3)
    # Regularise so degenerate (collinear / few-neighbour) nodes give R ~ I.
    S = S + eps * jnp.eye(3)[None]

    if rotation_method == "svd":
        R = _closest_rotation_svd(S)
    else:
        R = _closest_rotation_polar(S, polar_iters)

    # Optimal rotation is a target (ARAP local step): detach so we do not
    # backprop through the rotation solve (whose gradient is unstable at
    # degenerate spectra) and so the energy is a clean quadratic in the positions.
    R = jax.lax.stop_gradient(R)

    resid = e - jnp.einsum('eij,ej->ei', R[i], e0)              # (E, 3)
    per_edge = w * jnp.sum(resid ** 2., axis=-1)
    if weighted_mean:
        return jnp.sum(per_edge) / (jnp.sum(w) + eps)
    return jnp.mean(per_edge)


def calculate_deformation_coherence_loss(displacements, radius_graph, edge_weights, eps=1e-8):
    """Enforces smooth motion (Nearby points move together)."""
    i, j = radius_graph
    # Difference in the *change* of position
    diffs = displacements[i] - displacements[j]
    dist_sq = jnp.sum(diffs ** 2., axis=-1)

    return jnp.mean(edge_weights * dist_sq)

def calculate_repulsion_loss(positions, radius_graph, tau, edge_weights=None, eps=1e-8):
    """Prevents collisions/overlapping density"""
    i, j = radius_graph
    diffs = positions[i] - positions[j]
    distances = jnp.sqrt(jnp.sum(diffs ** 2., axis=-1) + eps)

    # Quadratic penalty if distance is less than tau (cutoff)
    cutoff = jnp.maximum(0.5, tau)
    # This acts like a 'soft' version of your multiplier trick
    penalty = jnp.clip(distances, a_max=cutoff)
    penalty = jnp.abs(penalty - cutoff)
    if edge_weights is not None:
        return jnp.sum(edge_weights * penalty) / (jnp.sum(edge_weights) + eps)
    return penalty.mean()

def calculate_outlier_loss(positions, knn_graph, tau, eps=1e-8):
    """Prevents Gaussians from detaching into the solvent."""
    i, j = knn_graph
    diffs = positions[i] - positions[j]
    distances = jnp.sqrt(jnp.sum(diffs ** 2., axis=-1) + eps)

    cutoff = 1.5 * tau
    penalty = jnp.clip(distances, a_min=cutoff)
    return jnp.mean((penalty - cutoff) ** 2.)

def calculate_neighbour_loss(positions, radius_graph, tau, num_points, eps=1e-8):
    """Maintains uniform density across the protein volume."""
    i, j = radius_graph

    # 1. Compute distances for graph edges
    diffs = positions[i] - positions[j]
    distances = jnp.sqrt(jnp.sum(diffs ** 2., axis=-1) + eps)

    # 2. Convert distances to a 'is_neighbor' score (0 to 1)
    dist_activation = _distance_activation(distances, tau)

    # 3. Sum scores for each point i (DynaMight uses scatter)
    # segment_sum expects i to be sorted or we use num_segments
    n_neighbours = jax.ops.segment_sum(
        dist_activation,
        segment_ids=i,
        num_segments=num_points
    )

    # 4. Penalize if the sum is outside [1, 3]
    neighbor_penalty = _neighbour_activation(n_neighbours, minimum=1.0, maximum=3.0)
    return jnp.mean(neighbor_penalty)


### KEEP FOR REFERENCE ###
# def distance_regularizer_from_graph(c0: jnp.ndarray,
#                                     c: jnp.ndarray,
#                                     edge_index: jnp.ndarray):
#     """
#     c0: (N, 3) reference centers (no grad needed).
#     c:  (N, 3) deformed centers (requires grad).
#     edge_index: (2, E) int array of edge indices [i, j].
#
#     Returns:
#         scalar loss (if mean/sum) or (E,) per-edge loss (if "none").
#     """
#     i, j = edge_index  # (E,), (E,)
#
#     d0 = safe_norm(c0[i] - c0[j], axis=-1)  # (E,)
#     d  = safe_norm(c[i]  - c[j],  axis=-1)  # (E,)
#
#     diff = d - d0
#     loss_per_edge = diff ** 2.  # (E,)
#
#     return jnp.mean(loss_per_edge)
#
#
# def repulsion_from_graph(c0: jnp.ndarray,
#                          c: jnp.ndarray,
#                          edge_index: jnp.ndarray):
#     """
#     c0: (N, 3) reference centers (no grad needed).
#     c:  (N, 3) deformed centers (requires grad).
#     edge_index: (2, E) int array of edge indices [i, j].
#
#     Returns:
#         scalar loss (if mean/sum) or (E,) per-edge loss (if "none").
#     """
#
#     i, j = edge_index  # (E,), (E,)
#
#     d  = safe_norm(c[i] - c[j],  axis=-1)  # (E,)
#
#     tau = safe_norm(c0[i] - c0[j], axis=-1).mean()
#
#     x = jnp.where(d < tau, 1.0, tau)
#
#     return jnp.mean(x * (d - tau) ** 2.).mean()


# --------------------------------------------------------------------------- #
#  Geometric correction loss                                                   #
# --------------------------------------------------------------------------- #
def _decoder_manifold_point(model, z):
    """Stack *all* latent-dependent decoder outputs for a single latent ``z``.

    The geometric correction loss measures the local volume scaling of the map
    ``D: R^d -> R^M`` from a latent ``z`` to the decoder output, via the Gram
    determinant of its Jacobian (see :func:`geometric_correction_loss`).

    Parameters
    ----------
    model : HetSIREN | Zernike3Deep | ReconSIREN | ReconSIRENHetOnly
    z     : (d,) array -- a single latent vector.

    Returns
    -------
    (M,) array -- the flattened, concatenated latent-dependent decoder outputs.
    """
    z_b = z[None, :]         # decoders expect a leading batch axis: (1, d)
    half = 0.5 * model.xsize  # voxel -> normalised box units

    if hasattr(model, "delta_het_decoder"):        # ReconSIREN, ReconSIRENHetOnly
        coords, values = model.delta_het_decoder(z_b)
        blocks = [(coords[0] / half).reshape(-1), values[0].reshape(-1)]
    elif hasattr(model, "delta_volume_decoder"):   # HetSIREN
        coords, values = model.delta_volume_decoder(z_b)
        blocks = [(coords[0] / half).reshape(-1), values[0].reshape(-1)]
    elif hasattr(model, "decode_field"):           # Zernike3Deep
        field = model.decode_field(z_b)[0]         # flow only; amplitudes are constant
        blocks = [field[0].reshape(-1)]
    else:
        raise TypeError(
            f"{type(model).__name__} exposes none of 'delta_het_decoder', "
            f"'delta_volume_decoder' or 'decode_field'; cannot locate its "
            f"heterogeneity decoder.")

    return jnp.concatenate(blocks)  # (M,)


def decoder_jacobian(latents, model):
    """Batched Jacobian ``J_D(z)`` of a model's heterogeneity decoder.

    Let ``D: R^d -> R^M`` be the map from a latent ``z`` to the stacked
    latent-dependent decoder outputs (see :func:`_decoder_manifold_point`). This
    returns the Jacobian of ``D`` evaluated at each latent in the batch.

    Forward-mode (:func:`jax.jacfwd`) is used because the output dimension ``M``
    (``~3N`` and more) is far larger than the latent dimension ``d`` (a handful),
    so ``d`` forward passes is much cheaper than ``M`` reverse passes.

    Parameters
    ----------
    latents : (B, d) array -- batch of latent vectors.
    model   : HetSIREN | Zernike3Deep | ReconSIREN | ReconSIRENHetOnly

    Returns
    -------
    (B, M, d) array -- ``jacobian[b, m, k] = d D(z_b)[m] / d z_b[k]`` where ``B``
        is the batch size, ``M`` the stacked output dimension and ``d`` the
        latent dimension.
    """
    single_jac = jax.jacfwd(lambda z: _decoder_manifold_point(model, z))  # (d,) -> (M, d)
    return jax.vmap(single_jac)(latents)                                  # (B, M, d)


def geometric_correction_loss(latents, model, eps=1e-8):
    """Geometric correction loss -- penalise local distortion of the latent map.

    Drives the decoder towards a *locally isometric* map by pushing the volume
    element ``sqrt(det(J^T J))`` (``J`` the decoder Jacobian) towards ``1`` at
    every latent.
    """
    # Decoder Jacobian over the stacked latent-dependent outputs
    jacobian = decoder_jacobian(latents, model)  # (B, M, d)

    # Gram matrix and its determinant (volume element squared)
    gram = jnp.swapaxes(jacobian, -1, -2) @ jacobian  # (B, d, d)
    gram_determinant = jnp.linalg.det(gram)           # (B,)

    # Geometric correction loss (clip guards against tiny negative dets)
    loss = jnp.square(1. - jnp.sqrt(jnp.clip(gram_determinant, a_min=0.0) + eps)).mean()

    return loss

def _subsample_indices(key, num_outputs, k):
    """``k`` distinct output (pixel) indices for the Monte-Carlo Gram estimator."""
    return jax.random.choice(key, num_outputs, shape=(k,), replace=False)


def _composed_gram(render_point_fn, z, ctx_i, num_outputs, pixel_subsample, key,
                   sequential):
    """``(d, d)`` Gram matrix ``A^T A`` of the composed map for one latent.

    ``render_point_fn(z, ctx_i)`` returns the flattened image ``(M=n^2,)`` for a
    single latent ``z`` and its per-particle physics context ``ctx_i`` (pose,
    CTF, ...), holding the physics constant w.r.t. ``z``.

    Two independent memory levers:

    * ``pixel_subsample`` (``k < M``): differentiate only ``k`` randomly chosen
      pixels and rescale by ``M / k``, so ``E[A_S^T A_S] = A^T A`` (an unbiased
      Johnson-Lindenstrauss / Hutchinson estimator of the *Gram matrix*; note the
      subsequent ``det`` is a nonlinear function of it, so ``det`` itself is only
      approximately recovered). Caps the stored Jacobian and the Gram product at
      ``O(k*d)`` instead of ``O(M*d)``.
    * ``sequential``: build the ``d`` Jacobian columns one at a time with
      :func:`jax.jvp` under :func:`jax.lax.scan`, instead of :func:`jax.jacfwd`
      which vectorises all ``d`` directions at once. This keeps the *renderer's*
      tangent memory at ``1x`` (a single forward image) rather than ``d x``,
      trading it for ``d`` sequential passes -- the lever that matters when
      ``n^2`` (not ``d``) dominates. Exact; composes with ``pixel_subsample``.
    """
    if pixel_subsample is not None and pixel_subsample < num_outputs:
        idx = _subsample_indices(key, num_outputs, pixel_subsample)
        scale = num_outputs / pixel_subsample
        f = lambda zz: render_point_fn(zz, ctx_i)[idx]          # (k,)
    else:
        scale = 1.0
        f = lambda zz: render_point_fn(zz, ctx_i)               # (M,)

    if sequential:
        # d sequential JVPs; each materialises one full tangent image (1x mem).
        eye = jnp.eye(z.shape[0], dtype=z.dtype)
        def column(carry, e):
            _, col = jax.jvp(f, (z,), (e,))                     # (m,)
            return carry, col
        _, cols = jax.lax.scan(column, None, eye)              # (d, m)
        A = cols.T                                             # (m, d)
    else:
        A = jax.jacfwd(f)(z)                                   # (m, d)

    return scale * (A.T @ A)                                   # (d, d)


def composed_geometric_correction_loss(latents, render_point_fn, ctx, num_outputs,
                                       *, pixel_subsample=None, key=None,
                                       sequential=False, eps=1e-8):
    """Geometric correction loss on the composed *decoder + physics* map.

    Pushes the volume element ``sqrt(det(J_F^T J_F))`` of ``F = Phi . D`` towards
    ``1`` at every latent, so the map from ``z`` to the *rendered image* is kept
    locally isometric -- the metric therefore includes the projection, Gaussian
    blur and CTF, not just the point-cloud deformation.

    Parameters
    ----------
    latents : (B, d) array
        Batch of latent vectors.
    render_point_fn : callable ``(z, ctx_i) -> (M,)``
        Renders the flattened image for a *single* latent ``z`` and its
        per-particle physics context ``ctx_i``. Must be differentiable in ``z``
        and treat ``ctx_i`` as constant. The caller builds this closure from its
        model's decoder + ``phys_decoder`` (the physics differs per model, so it
        lives at the call site, not here).
    ctx : pytree
        Per-particle physics context, batched over axis 0 (``ctx[b]`` feeds
        ``latents[b]``). Pass ``None`` if the renderer needs no context.
    num_outputs : int
        ``M = n*n``, the flattened image length (static; needed to rescale the
        subsampled estimator).
    pixel_subsample : int, optional
        If set, estimate each Gram matrix from ``k = pixel_subsample`` random
        pixels instead of all ``M`` -- ``O(k*d)`` memory instead of ``O(M*d)``.
    key : jax.Array, optional
        PRNG key for the pixel subsampling (required only when subsampling).
    sequential : bool, optional
        Build the Jacobian columns one at a time (``jvp`` + ``scan``) to hold the
        renderer's tangent memory at ``1x`` instead of ``d x``, at the cost of
        ``d`` sequential passes. Exact; combine with ``pixel_subsample`` for the
        smallest footprint. Default ``False`` (faster ``jacfwd``).

    Returns
    -------
    scalar -- the mean geometric correction loss over the batch.
    """
    B = latents.shape[0]
    if key is None:
        # Deterministic fallback so the function is usable without a key when no
        # subsampling is requested (the keys are then unused).
        keys = jnp.zeros((B, 2), dtype=jnp.uint32)
    else:
        keys = jax.random.split(key, B)

    def gram_one(z, ctx_i, k):
        return _composed_gram(render_point_fn, z, ctx_i, num_outputs,
                              pixel_subsample, k, sequential)

    grams = jax.vmap(gram_one)(latents, ctx, keys)     # (B, d, d)
    gram_determinant = jnp.linalg.det(grams)           # (B,)

    loss = jnp.square(1. - jnp.sqrt(jnp.clip(gram_determinant, a_min=0.0) + eps)).mean()
    return loss