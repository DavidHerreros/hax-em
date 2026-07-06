from functools import partial
import jax
from jax import numpy as jnp, lax as jlx
from jax.scipy.special import logsumexp
import numpy as np
import math
import chex
import dm_pix


def gradient_loss(s, penalty='l2'):
    dy = jnp.abs(s[:, :, 1:, :, :] - s[:, :, :-1, :, :])
    dx = jnp.abs(s[:, :, :, 1:, :] - s[:, :, :, :-1, :])
    dz = jnp.abs(s[:, :, :, :, 1:] - s[:, :, :, :, :-1])

    if (penalty == 'l2'):
        dy = dy * dy
        dx = dx * dx
        dz = dz * dz

    d = jnp.mean(dx) + jnp.mean(dy) + jnp.mean(dz)
    return d / 3.0


def diceLoss(y_true, y_pred):
    ndims = len(list(y_pred.size())) - 2
    vol_axes = list(range(2, ndims + 2))
    top = 2 * (y_true * y_pred).sum(axis=vol_axes)
    bottom = jnp.clip((y_true + y_pred).sum(axis=vol_axes), min=1e-5)
    dice = jnp.mean(top / bottom)
    return -dice


def compute_local_sums(I, J, win_spatial_dims, window_strides_arg, padding_values_arg, ndims):
    """
    Computes local sums for NCC calculation using JAX convolutions.
    This is a helper function for ncc_loss_jax.

    Args:
        I: Input JAX array with shape [batch_size, *vol_shape, nb_feats].
        J: Input JAX array with shape [batch_size, *vol_shape, nb_feats].
        win_spatial_dims: List of kernel sizes for spatial dimensions (e.g., [kH, kW] for 2D).
        window_strides_arg: Tuple of strides for each spatial dimension (e.g., (sH, sW)).
        padding_values_arg: Tuple of symmetric padding values for each spatial dimension (e.g., (pH, pW)).
        ndims: Number of spatial dimensions (1, 2, or 3).

    Returns:
        A tuple containing (I_var, J_var, cross_term), which are JAX arrays.
    """
    nb_feats = I.shape[-1]

    # Basic input validation
    if I.shape[:-1] != J.shape[:-1] or I.shape[-1] != J.shape[-1]:
        raise ValueError(f"Input arrays I and J must have the same dimensions. Got I: {I.shape}, J: {J.shape}")
    if I.ndim != ndims + 2:  # batch_size, *vol_shape (ndims), nb_feats
        raise ValueError(f"Input array I has incorrect number of dimensions ({I.ndim}) for ndims={ndims}.")
    if J.ndim != ndims + 2:
        raise ValueError(f"Input array J has incorrect number of dimensions ({J.ndim}) for ndims={ndims}.")
    if nb_feats == 0 and I.size > 0:  # Check I.size to avoid error on genuinely empty input
        raise ValueError("Input tensor I has 0 features (channels) but is not an empty array.")

    # Kernel for jax.lax.conv_general_dilated
    # Shape: (*kernel_spatial_dims, C_in_per_group, C_out_per_group)
    # We want to sum each feature independently using the same windowed filter.
    # So, C_in_per_group = 1 (filter takes one channel from the group at a time)
    # C_out_per_group = 1 (filter produces one channel for that group)
    # feature_group_count will be nb_feats (each input feature is its own group).
    kernel_spatial_shape = tuple(win_spatial_dims)
    sum_filt_kernel = jnp.ones(kernel_spatial_shape + (1, 1), dtype=I.dtype)

    # Strides for jax.lax.conv_general_dilated (e.g., (1,) or (1,1) or (1,1,1))
    strides_jax = window_strides_arg

    # Padding for jax.lax.conv_general_dilated
    # Needs to be a list of (pad_low, pad_high) pairs for each spatial dimension.
    padding_jax = []
    for i in range(ndims):
        # padding_values_arg contains symmetric padding amount for each dim, e.g., (pad_H, pad_W) for 2D
        pad_val = padding_values_arg[i]
        padding_jax.append((pad_val, pad_val))  # Symmetric padding

    # Set dimension_numbers for jax.lax.conv_general_dilated
    # lhs (input): (N, *Spatial, C) - N: batch, C: channels/features
    # rhs (kernel): (*KernelSpatial, I_group, O_group) - I: input feats/group, O: output feats/group
    # out (output): (N, *Spatial_out, C_total)
    if ndims == 1:
        dimension_numbers = ('NWC', 'WIO', 'NWC')  # W: width
    elif ndims == 2:
        dimension_numbers = ('NHWC', 'HWIO', 'NHWC')  # H: height, W: width
    elif ndims == 3:
        dimension_numbers = ('NDHWC', 'DHWIO', 'NDHWC')  # D: depth, H: height, W: width
    else:
        raise ValueError(f"Unsupported number of dimensions (ndims): {ndims}. Must be 1, 2, or 3.")

    def convolve(data_lhs):
        """Applies the convolution to sum values in a local window."""
        return jlx.conv_general_dilated(
            lhs=data_lhs,  # Input data
            rhs=sum_filt_kernel,  # Kernel (filter)
            window_strides=strides_jax,  # Strides for the convolution
            padding=padding_jax,  # Padding for each spatial dimension
            dimension_numbers=dimension_numbers,  # Specifies layout of dims
            feature_group_count=nb_feats  # Applies sum_filt_kernel to each input feature independently
        )

    # Calculate squared values and product
    I2 = I * I
    J2 = J * J
    IJ = I * J

    # Compute local sums using convolution
    I_sum = convolve(I)
    J_sum = convolve(J)
    I2_sum = convolve(I2)
    J2_sum = convolve(J2)
    IJ_sum = convolve(IJ)

    # Calculate window size (number of elements in the window)
    win_size_float = float(np.prod(win_spatial_dims))
    if win_size_float == 0:  # Should not happen with valid win_spatial_dims
        raise ValueError("Window size (product of win_spatial_dims) is zero.")

    # Local means
    u_I = I_sum / win_size_float
    u_J = J_sum / win_size_float

    # Local variance and cross-correlation terms
    # cross = sum_window((I - u_I)(J - u_J))
    cross_term = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size_float
    # I_var = sum_window((I - u_I)^2)
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size_float
    # J_var = sum_window((J - u_J)^2)
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size_float

    return I_var, J_var, cross_term

def ncc_loss(I, J, win_arg=None):
    """
    Calculates the Normalized Cross-Correlation (NCC) loss between I and J using JAX.

    The NCC is a measure of similarity between two images or signals. This loss
    is -NCC^2, meaning it encourages high magnitudes of correlation (either positive or negative).

    Args:
        I: Input JAX array with shape [batch_size, *vol_shape, nb_feats].
           Assumes channel-last format (e.g., NHWC for 2D images).
        J: Input JAX array with the same shape and format as I.
        win_arg: Optional. Window size for local NCC calculation.
                 - If None, defaults to [9]*ndims (e.g., 9x9 for 2D).
                 - If an int, used for all spatial dimensions (e.g., win_arg=7 means 7x7 for 2D).
                 - If a list/tuple, specifies window size for each spatial dimension
                   (e.g., win_arg=[7,5] for 2D HxW). Length must match ndims.

    Returns:
        A scalar JAX array representing the mean NCC loss (-NCC^2).
    """
    if not isinstance(I, jnp.ndarray) or not isinstance(J, jnp.ndarray):
        raise TypeError("Inputs I and J must be JAX numpy arrays.")

    # Determine number of spatial dimensions
    ndims = I.ndim - 2  # (batch_dim + feature_dim)
    if not (1 <= ndims <= 3):
        raise ValueError(
            f"Input volume dimensionality (ndims={ndims}, derived from I.ndim={I.ndim}) must be 1, 2, or 3."
        )

    # Process window argument
    if win_arg is None:
        win_spatial_dims = [9] * ndims  # Default window size
    else:
        if isinstance(win_arg, int):
            win_spatial_dims = [win_arg] * ndims
        elif isinstance(win_arg, (list, tuple)):
            if len(win_arg) != ndims:
                raise ValueError(
                    f"Length of win_arg ({len(win_arg)}) must match number of spatial dimensions ({ndims})."
                )
            win_spatial_dims = list(win_arg)
        else:
            raise TypeError("win_arg must be None, an int, or a list/tuple of ints.")

    # Validate window dimensions
    for k_dim_size in win_spatial_dims:
        if not isinstance(k_dim_size, int) or k_dim_size <= 0:
            raise ValueError("Window dimensions in win_arg must be positive integers.")

    # Determine padding for each spatial dimension (symmetric padding)
    # This ensures the output of convolution has the same spatial dimensions as input if stride is 1.
    pad_values_per_dim = []
    for k_dim_size in win_spatial_dims:
        pad_values_per_dim.append(math.floor(k_dim_size / 2))

    # Strides and padding tuples based on ndims, to be passed to compute_local_sums_jax
    # Strides are typically (1,) for 1D, (1,1) for 2D, (1,1,1) for 3D for standard NCC.
    # padding_arg_tuple for compute_local_sums_jax is (pad_dim1, pad_dim2, ...)
    if ndims == 1:
        stride_tuple = (1,)
        padding_arg_tuple = (pad_values_per_dim[0],)
    elif ndims == 2:
        stride_tuple = (1, 1)
        padding_arg_tuple = (pad_values_per_dim[0], pad_values_per_dim[1])
    else:  # ndims == 3
        stride_tuple = (1, 1, 1)
        padding_arg_tuple = (pad_values_per_dim[0], pad_values_per_dim[1], pad_values_per_dim[2])

    # Compute local variances and cross-correlation term
    I_var, J_var, cross = compute_local_sums(
        I, J, win_spatial_dims, stride_tuple, padding_arg_tuple, ndims
    )

    # Calculate squared NCC: (Cov(I,J)^2) / (Var(I) * Var(J))
    # Note: 'cross' term is N * Cov(I,J), and 'I_var'/'J_var' are N * Var(I)/Var(J)
    # So, (N*Cov)^2 / ((N*Var_I)*(N*Var_J)) = Cov^2 / (Var_I * Var_J), which is NCC_squared.
    # Adding a small epsilon to the denominator for numerical stability.
    ncc_squared = (cross * cross) / (I_var * J_var + 1e-5)

    # The loss is the negative mean of the squared NCC.
    # This encourages high correlation magnitude (positive or negative).
    loss = -1 * jnp.mean(ncc_squared)
    return loss

def correlation_coefficient_loss(x, y):
    epsilon = 10e-5
    mx = jnp.mean(x, axis=[1, 2], keepdims=True)
    my = jnp.mean(y, axis=[1, 2], keepdims=True)
    xm, ym = x - mx, y - my
    r_num = jnp.sum(xm * ym, axis=[1, 2])
    x_square_sum = jnp.sum(xm * xm, axis=[1, 2])
    y_square_sum = jnp.sum(ym * ym, axis=[1, 2])
    r_den = jnp.sqrt(x_square_sum * y_square_sum)
    r = r_num / (r_den + epsilon)
    return jnp.mean(1. - r, axis=-1)

def simae(
    a: chex.Array,
    b: chex.Array,
    *,
    ignore_nans: bool = False,
) -> chex.Numeric:
  """Returns the Scale-Invariant Mean Squared Error between `a` and `b`.

  For each image pair, a scaling factor for `b` is computed as the solution to
  the following problem:

    min_alpha || vec(a) - alpha * vec(b) ||_2^2

  where `a` and `b` are flattened, i.e., vec(x) = np.flatten(x). The MSE between
  the optimally scaled `b` and `a` is returned: mse(a, alpha*b).

  This is a scale-invariant metric, so for example: simse(x, y) == sims(x, y*5).

  This metric was used in "Shape, Illumination, and Reflectance from Shading" by
  Barron and Malik, TPAMI, '15.

  Args:
    a: First image (or set of images).
    b: Second image (or set of images).
    ignore_nans: If True, will ignore NaNs in the inputs.

  Returns:
    SIMAE between `a` and `b`.
  """
  # DO NOT REMOVE - Logging usage.

  chex.assert_rank([a, b], {3, 4})
  chex.assert_type([a, b], float)
  chex.assert_equal_shape([a, b])

  sum_fn = jnp.nansum if ignore_nans else jnp.sum
  a_dot_b = sum_fn((a * b), axis=(-3, -2, -1), keepdims=True)
  b_dot_b = sum_fn((b * b), axis=(-3, -2, -1), keepdims=True)
  alpha = a_dot_b / b_dot_b
  return dm_pix.mae(a, alpha * b, ignore_nans=ignore_nans)

def contrastive_ce_loss(
    dist_pos: jnp.ndarray,
    dist_neg: jnp.ndarray,
    temperature: float = 0.07,
    reduction: str = "mean",
):
    """
    InfoNCE‑style contrastive loss using pre‑computed distance matrices.

    Parameters
    ----------
    dist_pos : (M, P) array
        Euclidean (or other) distances from each of M anchors
        to their P *closest / similar* neighbours.
    dist_neg : (M, P) array
        Distances from the same anchors to their P *farthest / dissimilar* neighbours.
    temperature : float, default 0.07
        Soft‑max temperature τ used in SimCLR, MoCo, etc.
    reduction : {'mean', 'sum', 'none'}, default 'mean'
        Aggregation mode applied over all (anchor, positive) pairs.

    Returns
    -------
    jnp.ndarray
        • scalar loss if reduction is 'mean' or 'sum'
        • (M, P) array of individual losses if reduction == 'none'
    """
    if dist_pos.shape != dist_neg.shape:
        raise ValueError("dist_pos and dist_neg must have the same shape")

    # ------------------------------------------------------------------
    # 1.  Convert **distance** to **similarity**: s = −d
    #     (smaller distances → larger similarities).
    # 2.  Scale by temperature τ.
    # ------------------------------------------------------------------
    pos_logits = -dist_pos / temperature          # (M, P)
    neg_logits = -dist_neg / temperature          # (M, P)

    # ------------------------------------------------------------------
    # 3.  Compute log‑denominator        log( e^{s⁺/τ} + ∑ e^{s⁻/τ} )
    #     for every (anchor, positive) pair in a numerically stable way.
    # ------------------------------------------------------------------
    # log ∑ e^{s⁻/τ}   – one value per anchor, shape (M, 1)
    neg_lse = logsumexp(neg_logits, axis=1, keepdims=True)

    # log( e^{s⁺/τ} + ∑ e^{s⁻/τ} )  – broadcasts over the P positives
    log_denom = jnp.logaddexp(pos_logits, neg_lse)

    # ------------------------------------------------------------------
    # 4.  InfoNCE loss  −s⁺/τ + log‑denominator  (per positive sample)
    # ------------------------------------------------------------------
    loss_per_pair = -pos_logits + log_denom       # (M, P)

    # ------------------------------------------------------------------
    # 5.  Reduction
    # ------------------------------------------------------------------
    if reduction == "mean":
        return jnp.mean(loss_per_pair)
    if reduction == "sum":
        return jnp.sum(loss_per_pair)
    if reduction == "none":
        return loss_per_pair
    raise ValueError("reduction must be 'mean', 'sum' or 'none'")

def triplet_loss(
    dist_pos: jnp.ndarray,
    dist_neg: jnp.ndarray,
    margin: float = 1.0,
    reduction: str = "mean",
):
    """
    Triplet loss for pre‑computed distance matrices.

    Parameters
    ----------
    dist_pos : (M, P) array
        Distances from each of M anchors to their P *closest* (similar) neighbours.
    dist_neg : (M, P) array
        Distances from each of M anchors to their P *farthest* (dissimilar) neighbours.
    margin : float, default 1.0
        Desired minimum distance between dissimilar pairs.
    reduction : {'mean', 'sum', 'none'}, default 'mean'
        How to aggregate the per‑pair losses.

    Returns
    -------
    jnp.ndarray
        • scalar loss if reduction is 'mean' or 'sum'
        • (M, P) array of individual losses if reduction == 'none'
    """
    if dist_pos.shape != dist_neg.shape:
        raise ValueError("dist_pos and dist_neg must have the same shape")

    # Positive term ‑‑ pull similar samples together
    # loss_pos = jnp.square(dist_pos)
    loss_pos = dist_pos

    # Negative term ‑‑ push dissimilar samples apart (only if within margin)
    # loss_neg = jnp.square(jnp.clip(margin - dist_neg, a_min=0.0))
    loss_neg = jnp.clip(margin - dist_neg, a_min=0.0)

    per_pair = 0.5 * (loss_pos + loss_neg)   # 0.5 is conventional; optional

    if reduction == "mean":
        return jnp.mean(per_pair)
    if reduction == "sum":
        return jnp.sum(per_pair)
    if reduction == "none":
        return per_pair
    raise ValueError("reduction must be 'mean', 'sum' or 'none'")


@partial(jax.jit, static_argnames=['num_projections',])
def sliced_wasserstein_loss(x: jnp.ndarray, x_true: jnp.ndarray, key: jax.random.PRNGKey, num_projections: int = 128) -> jnp.ndarray:
    """
    Computes the Sliced-Wasserstein-2 distance to a uniform distribution.

    Args:
        x: Input array of shape (N, 3).
        x_true: True data of shape (N, 3).
        num_projections: The number of random 1D projections to use.

    Returns:
        A scalar loss value.
    """
    N, D = x.shape

    # 1. Generate random projections
    key, proj_key, true_key = jax.random.split(key, 3)
    projections = jax.random.normal(proj_key, shape=(D, num_projections))
    projections = projections / jnp.linalg.norm(projections, axis=0, keepdims=True)

    # 2. Project both the input data and the true uniform data
    x_proj = x @ projections
    x_true_proj = x_true @ projections

    # 3. Sort the projections along the N-axis
    x_proj_sorted = jnp.sort(x_proj, axis=0)
    x_true_proj_sorted = jnp.sort(x_true_proj, axis=0)

    # 4. Compute the L2 distance between sorted projections and average over all projections
    # This is the squared Sliced-Wasserstein-2 distance
    loss = jnp.mean((x_proj_sorted - x_true_proj_sorted) ** 2)

    return loss


def build_fourier_rings(box_size: int) -> tuple[jax.Array, int]:
    """Build a one-hot ring membership tensor for an rFFT.

    Parameters
    ----------
    box_size
        Image side length in pixels.

    Returns
    -------
    rings
        ``(H, W//2+1, n_rings)`` float32 tensor.  ``rings[y, x, r]`` is
        1.0 if Fourier coefficient ``(y, x)`` belongs to ring ``r``,
        else 0.0.
    n_rings
        Number of rings = ``box_size // 2 + 1`` (DC through Nyquist).

    The radius assignment matches the original ``build_fourier_rings``
    exactly: integer-rounded Euclidean radius from DC, with coefficients
    beyond Nyquist excluded.
    """
    H = box_size
    W = box_size // 2 + 1
    nyquist = box_size // 2
    n_rings = nyquist + 1

    fx = np.arange(W)
    fy = np.where(np.arange(H) <= H // 2,
                  np.arange(H),
                  np.arange(H) - H)

    r = np.sqrt(fx[None, :] ** 2 + fy[:, None] ** 2)
    ring_idx = np.round(r).astype(np.int32)

    # Build one-hot: rings[y, x, r] = 1 iff ring_idx[y, x] == r AND r <= nyquist.
    # Indices > nyquist are automatically excluded because we only allocate
    # n_rings slots and use np.where to clip.
    rings_np = np.zeros((H, W, n_rings), dtype=np.float32)
    valid = ring_idx <= nyquist
    yy, xx = np.where(valid)
    rings_np[yy, xx, ring_idx[yy, xx]] = 1.0

    return jnp.asarray(rings_np), n_rings


def preprocess_particles(
        images: jax.Array,
        apply_mean_subtract: bool = True,
) -> jax.Array:
    """Convert real particle images to their rFFT for FRC computation."""
    if apply_mean_subtract:
        images = images - images.mean(axis=(-2, -1), keepdims=True)
    return jnp.fft.rfft2(images, norm="ortho")


def frc_loss(
        pred_ft: jax.Array,
        obs_ft: jax.Array,
        rings: jax.Array,
        band_mask: jax.Array,
        eps: float = 1e-3,
) -> jax.Array:
    """Compute the negative mean FRC over the precomputed band.

    Parameters
    ----------
    pred_ft, obs_ft
        Complex (B, H, W//2+1) rFFT tensors.
    rings
        One-hot ring tensor (H, W//2+1, n_rings) from
        ``build_fourier_rings``.
    band_mask
        Precomputed (n_rings,) float mask: 1.0 inside the band, 0.0
        outside.  Built once at ``FRCLoss`` construction time.
    eps
        Relative floor on the per-ring power (as a fraction of the peak observed
        ring power) used to stabilize the denominator. Prevents the FRC gradient
        from exploding when a ring has near-zero predicted power, which would
        otherwise drive downstream coordinates to NaN. Scale-invariant.

    Returns
    -------
    scalar loss = -mean over batch of (mean over band of FRC).
    """
    # |pred_ft|^2 and |obs_ft|^2 via real/imag parts — direct, no complex mul.
    pred_r, pred_i = pred_ft.real, pred_ft.imag
    obs_r, obs_i = obs_ft.real, obs_ft.imag

    pred_sq = pred_r * pred_r + pred_i * pred_i  # (B, H, W//2+1)
    obs_sq = obs_r * obs_r + obs_i * obs_i  # (B, H, W//2+1)
    cross = pred_r * obs_r + pred_i * obs_i  # (B, H, W//2+1) — Re(pred · conj(obs))

    # Stack the three reductions, do one tensordot.
    # stacked: (3, B, H, W//2+1) → contract axes (2,3) with rings axes (0,1).
    stacked = jnp.stack([cross, pred_sq, obs_sq], axis=0)
    # Result: (3, B, n_rings)
    reduced = jnp.tensordot(stacked, rings, axes=[[2, 3], [0, 1]])

    cross_r = reduced[0]
    pred_sq_r = reduced[1]
    obs_sq_r = reduced[2]

    # Per-ring FRC
    floor = eps * jnp.max(obs_sq_r, axis=-1, keepdims=True)
    denom = jnp.sqrt(jnp.maximum(pred_sq_r, floor) * jnp.maximum(obs_sq_r, floor)) + 1e-12
    frc = jnp.clip(cross_r / denom, -1.0, 1.0)  # (B, n_rings)

    # Mean over band, then mean over batch
    band_count = jnp.maximum(band_mask.sum(), 1.0)
    per_batch_frc = (frc * band_mask[None, :]).sum(axis=1) / band_count
    return -jnp.mean(per_batch_frc)


class FRCLoss:
    """FRC loss wrapper to simplify its call.

    Internally, computed the optimal bands and stores them so they don't need to be passed during the call
    """

    def __init__(
            self,
            box_size: int,
            apix: float,
            min_resolution_A: float = 30.0,
            max_resolution_A: float = 8.0,
            apply_mean_subtract: bool = True,
    ):
        minpx, maxpx = recommended_band(
            box_size=box_size, apix=apix,
            min_resolution_A=min_resolution_A,
            max_resolution_A=max_resolution_A,
        )

        self.box_size = box_size
        self.minpx = minpx
        self.maxpx = maxpx
        self.apply_mean_subtract = apply_mean_subtract
        self.rings, self.n_rings = build_fourier_rings(box_size)

        assert 0 <= minpx < maxpx < self.n_rings, (
            f"minpx={minpx}, maxpx={maxpx} must satisfy "
            f"0 <= minpx < maxpx < {self.n_rings}"
        )

        # Precompute the band mask once — same closed-closed semantics
        # [minpx, maxpx] as the original (using <= on both ends).
        self.ring_ids = jnp.arange(self.n_rings)
        band_mask_np = ((np.arange(self.n_rings) >= minpx) & (np.arange(self.n_rings) <= maxpx)).astype(np.float32)
        self.band_mask = jnp.asarray(band_mask_np)

    def _band_mask(self, freq_alpha, ceiling_px=None):
        """Band mask, optionally annealed and FSC-capped.

        ``freq_alpha`` in [0, 1] ramps the upper ring from ``minpx`` (coarse) up
        to ``maxpx`` (full band). ``ceiling_px`` (if given) additionally caps the
        upper ring at an FSC-derived resolution. Both accept traced scalars so
        this works inside ``jit``.
        """
        return dynamic_band_mask(self.ring_ids, self.minpx, self.maxpx, freq_alpha, ceiling_px)

    def __call__(self, pred_real: jax.Array, obs_real: jax.Array, freq_alpha=1.0, ceiling_px=None) -> jax.Array:
        pred_ft = preprocess_particles(pred_real,
                                       apply_mean_subtract=self.apply_mean_subtract)
        obs_ft = preprocess_particles(obs_real,
                                      apply_mean_subtract=self.apply_mean_subtract)
        return self.call_complex(pred_ft, obs_ft, freq_alpha, ceiling_px)

    def call_complex(self, pred_ft: jax.Array, obs_ft: jax.Array, freq_alpha=1.0, ceiling_px=None) -> jax.Array:
        return frc_loss(pred_ft, obs_ft, self.rings, self._band_mask(freq_alpha, ceiling_px))


def recommended_band(
        box_size: int,
        apix: float,
        min_resolution_A: float = 30.0,
        max_resolution_A: float = 8.0,
) -> tuple[int, int]:
    minpx = int(round(box_size * apix / min_resolution_A))
    maxpx = int(round(box_size * apix / max_resolution_A))
    minpx = max(1, minpx)
    maxpx = min(box_size // 2 - 1, maxpx)
    assert minpx < maxpx, (
        f"computed minpx={minpx} >= maxpx={maxpx}; check inputs"
    )
    return minpx, maxpx


def dynamic_band_mask(ring_ids: jax.Array, minpx: int, maxpx: int, freq_alpha=1.0, ceiling_px=None) -> jax.Array:
    """Soft band mask with an annealed and (optionally) FSC-capped upper limit.

    Rings below ``minpx`` are always excluded. The upper limit ramps linearly
    from ``minpx`` (``freq_alpha=0``) up to ``maxpx`` (``freq_alpha=1``), with a
    1-ring-wide soft edge so the newly admitted shell fades in smoothly instead
    of switching on abruptly (which would otherwise cause a loss/grad jump).

    ``ceiling_px`` (if given) is an additional hard cap on the upper ring, meant
    to be driven by a gold-standard half-set FSC resolution so the loss never
    tries to fit beyond the resolution the data actually supports. The effective
    upper ring is ``min(freq_alpha-ramp, ceiling_px)``. Both ``freq_alpha`` and
    ``ceiling_px`` may be traced scalars, so this is ``jit``-safe.

    With ``freq_alpha=1.0`` and ``ceiling_px=None`` (or >= maxpx) this reproduces
    the closed-closed ``[minpx, maxpx]`` hard mask exactly.
    """
    freq_alpha = jnp.clip(jnp.asarray(freq_alpha, dtype=jnp.float32), 0.0, 1.0)
    cur_max = minpx + freq_alpha * (maxpx - minpx)
    if ceiling_px is not None:
        cur_max = jnp.minimum(cur_max, jnp.asarray(ceiling_px, dtype=jnp.float32))
    lower = (ring_ids >= minpx).astype(jnp.float32)
    # Soft upper edge: 1 well inside the band, linearly to 0 one ring past cur_max.
    upper = jnp.clip(cur_max - ring_ids.astype(jnp.float32) + 1.0, 0.0, 1.0)
    return lower * upper


def build_fourier_shells_3d(box_size: int) -> tuple[jax.Array, int]:
    """Integer radial-shell index for a full (unshifted) 3D FFT grid.

    Returns ``(shell_index, n_shells)`` where ``shell_index`` is an
    ``(D, D, D)`` int32 array giving the rounded frequency radius (in pixels,
    0..Nyquist) of every voxel of ``jnp.fft.fftn`` output, and
    ``n_shells = box_size // 2 + 1`` (DC..Nyquist). Voxels beyond Nyquist (the
    corners) are clamped to the last shell; treat that last shell as unreliable.
    """
    f = jnp.fft.fftfreq(box_size) * box_size            # -D/2 .. D/2-1, in pixels
    kz, ky, kx = jnp.meshgrid(f, f, f, indexing="ij")
    r = jnp.sqrt(kx ** 2 + ky ** 2 + kz ** 2)
    n_shells = box_size // 2 + 1
    shell = jnp.clip(jnp.round(r).astype(jnp.int32), 0, n_shells - 1)
    return shell, n_shells


def fsc_3d(vol1: jax.Array, vol2: jax.Array, shell_index: jax.Array, n_shells: int,
           mask: jax.Array = None) -> jax.Array:
    """Fourier Shell Correlation between two 3D volumes.

    ``FSC(s) = Σ Re(F1·conj(F2)) / sqrt(Σ|F1|²·Σ|F2|²)`` per shell — the 3D
    analogue of :func:`frc_loss`'s per-ring correlation. Pass ``shell_index`` /
    ``n_shells`` from :func:`build_fourier_shells_3d`. An optional real-space
    ``mask`` (e.g. the reconstruction mask) is applied before the FFT so solvent
    does not inflate the correlation.

    Returns an ``(n_shells,)`` array (DC..Nyquist).
    """
    if mask is not None:
        vol1 = vol1 * mask
        vol2 = vol2 * mask
    f1 = jnp.fft.fftn(vol1)
    f2 = jnp.fft.fftn(vol2)
    cross = (f1.real * f2.real + f1.imag * f2.imag).reshape(-1)
    p1 = (f1.real ** 2 + f1.imag ** 2).reshape(-1)
    p2 = (f2.real ** 2 + f2.imag ** 2).reshape(-1)
    seg = shell_index.reshape(-1)
    cross_s = jax.ops.segment_sum(cross, seg, num_segments=n_shells)
    p1_s = jax.ops.segment_sum(p1, seg, num_segments=n_shells)
    p2_s = jax.ops.segment_sum(p2, seg, num_segments=n_shells)
    return cross_s / (jnp.sqrt(p1_s * p2_s) + 1e-12)


def fsc_resolution(fsc: jax.Array, box_size: int, apix: float, threshold: float = 0.5) -> tuple:
    """Resolution (Å) and shell index where the FSC first drops below ``threshold``.

    Uses ``threshold=0.5`` by default (the conservative "half-bit"-style
    criterion), which is deliberately stricter than the 0.143 gold-standard
    criterion so a decoded-map FSC — which is known to be optimistic — does not
    drive the loss band too far. Shell 0 (DC) is skipped. If the FSC never falls
    below the threshold, returns the Nyquist shell.

    Returns ``(resolution_A, shell)`` as host ints/floats.
    """
    fsc = np.asarray(fsc)
    n = fsc.shape[0]
    below = fsc[1:] < threshold
    if np.any(below):
        shell = int(np.argmax(below)) + 1
    else:
        shell = n - 1
    resolution_A = box_size * apix / max(shell, 1)
    return float(resolution_A), int(shell)


def conservative_band_ceiling(resolution_A: float, box_size: int, apix: float,
                              minpx: int, maxpx: int, margin: float = 1.15,
                              prev_ceiling_px: int = None, max_step: int = 2) -> int:
    """Turn an FSC resolution into a conservative upper-band ring for the loss.

    Three guards against an over-optimistic decoded-map FSC:
      * ``margin`` (>1) coarsens the resolution before converting to a ring, so
        the band stays a shell or two inside the measured FSC limit;
      * the result is clamped to ``[minpx+1, maxpx]`` (never below the low-res
        floor, never past the target band);
      * ``max_step`` + ``prev_ceiling_px`` add hysteresis: the ceiling may only
        rise a few rings per update, so a single optimistic FSC spike cannot
        immediately open the band.

    Returns an int ring index to pass as ``ceiling_px`` to the loss.
    """
    res = resolution_A * margin
    ceil_px = int(round(box_size * apix / max(res, 1e-6)))
    ceil_px = int(np.clip(ceil_px, minpx + 1, maxpx))
    if prev_ceiling_px is not None:
        ceil_px = min(ceil_px, int(prev_ceiling_px) + max_step)
        ceil_px = int(np.clip(ceil_px, minpx + 1, maxpx))
    return ceil_px


def chamfer_distance(x, y):
    # x: (N, D), y: (M, D) — N and M can differ
    diff = x[:, None, :] - y[None, :, :]      # (N, M, D)
    d2 = jnp.sum(diff ** 2, axis=-1)          # (N, M) squared distances
    x_to_y = jnp.min(d2, axis=1)              # (N,) nearest y for each x
    y_to_x = jnp.min(d2, axis=0)              # (M,) nearest x for each y
    return jnp.mean(x_to_y) + jnp.mean(y_to_x)
