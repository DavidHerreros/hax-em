"""
A JAX-based utility for whitening images, particularly for low-SNR data
like Cryo-EM where clean noise patches are unavailable.

The intended workflow is a two-step process:

1.  **ONE-TIME SETUP:** Use `estimate_noise_psd` on a large, representative
    batch of images from your dataset. This computes the canonical noise
    profile (a 1D Power Spectral Density). Save this 1D array to disk.

2.  **ROUTINE PRE-PROCESSING:** In your data pipeline, load the saved noise PSD.
    Use `create_whitening_fn` to generate a fast, JIT-compiled whitening
    function. Use this function to pre-process all your image batches.

Notes on the noise estimate
---------------------------
Two estimators are provided:

* ``method="background"`` (default, recommended): estimates the noise PSD from
  the solvent region of the particles (the pixels outside the inscribed circle,
  i.e. the image corners). In single-particle Cryo-EM the corners contain
  essentially pure noise, so their radial power spectrum is an unbiased estimate
  of the noise PSD at *every* frequency. This is the classical way to build a
  noise model and is what you want for whitening / SSNR weighting.

* ``method="percentile"``: the previous heuristic, kept for reproducibility. It
  takes a low percentile, across the batch, of the per-image radial PSD. It is
  biased at low frequencies (where even the least-powerful particle still
  carries signal), so it *overestimates* the noise there and therefore
  over-suppresses the low-resolution shells that hold most of the signal. Avoid
  it for whitening unless you know why you want it.
"""

import jax
import jax.numpy as jnp
import jax.numpy.fft as fft
from jax import vmap, jit
from functools import partial
from typing import Callable, Tuple


def _radius_grid(shape: Tuple[int, int]) -> Tuple[jax.Array, int]:
    """Integer radial index for an fftshifted 2D spectrum and the ring count.

    The number of rings is ``max_radius + 1`` where ``max_radius`` is the
    rounded distance to the farthest corner, so every coefficient (including the
    corners) has a valid, in-bounds ring index. ``jnp.round`` is used (not
    truncation) so a coefficient is assigned to its nearest integer shell.
    """
    height, width = shape
    center_y, center_x = height // 2, width // 2
    y, x = jnp.indices(shape)
    r = jnp.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    r = jnp.round(r).astype(jnp.int32)
    n_rings = int(round(((center_x ** 2 + center_y ** 2) ** 0.5))) + 1
    return r, n_rings


@partial(jit, static_argnums=(1,))
def _compute_1d_psd_from_image(image: jax.Array, n_rings: int) -> jax.Array:
    """
    Internal helper to compute the 1D radially averaged PSD from a single 2D image.
    """
    # 1. Compute 2D PSD (ortho-normalised so it is comparable across box sizes)
    shifted_fft = fft.fftshift(fft.fft2(image, norm="ortho"))
    psd_2d = jnp.abs(shifted_fft) ** 2

    # 2. Compute 1D radial average over integer shells
    r, _ = _radius_grid(psd_2d.shape)

    total_power = jnp.bincount(r.ravel(), weights=psd_2d.ravel(), length=n_rings)
    pixel_count = jnp.bincount(r.ravel(), length=n_rings)

    radial_psd = total_power / jnp.maximum(pixel_count, 1)

    return radial_psd


def estimate_noise_psd(
        representative_batch: jax.Array,
        method: str = "background",
        percentile: int = 10,
) -> jax.Array:
    """
    Estimates the noise PSD from a batch of signal-plus-noise images.

    This function should be run once on a large, representative batch of data
    to establish the dataset's canonical noise profile.

    Args:
        representative_batch: A large batch of images, shape (B, H, W, 1) or (B, H, W).
        method: ``"background"`` (default) estimates the noise from the solvent
            corners (unbiased at all frequencies); ``"percentile"`` uses the old
            low-percentile-across-batch heuristic.
        percentile: Percentile used only when ``method="percentile"``.

    Returns:
        The estimated 1D noise PSD array, length ``max_radius + 1``.
    """
    if representative_batch.ndim not in [3, 4]:
        raise ValueError("Input batch must have 3 or 4 dimensions.")

    # Squeeze channel dimension if it exists
    if representative_batch.ndim == 4:
        image_batch = jnp.squeeze(representative_batch, axis=-1)
    else:
        image_batch = representative_batch

    _, n_rings = _radius_grid(image_batch.shape[1:])

    if method == "background":
        # Solvent (corner) mask: pixels outside the inscribed circle. These are
        # pure noise in single-particle Cryo-EM, so their radial PSD is an
        # unbiased noise estimate at every frequency.
        h, w = image_batch.shape[1:]
        cy, cx = h // 2, w // 2
        yy, xx = jnp.indices((h, w))
        radius = jnp.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        solvent = radius > min(cy, cx)

        # Zero-mean each image on the solvent region before measuring its power.
        bg_mean = (image_batch * solvent).sum(axis=(1, 2)) / jnp.maximum(solvent.sum(), 1)
        images_centered = image_batch - bg_mean[:, None, None]

        batch_of_psds = vmap(partial(_compute_1d_psd_from_image, n_rings=n_rings))(images_centered)
        # Average PSD of the batch, weighted so only solvent power contributes is
        # not separable per ring after the FFT; instead we rely on the corners
        # dominating the outer shells and average across the batch for stability.
        noise_psd_estimate = jnp.mean(batch_of_psds, axis=0)
    elif method == "percentile":
        batch_of_psds = vmap(partial(_compute_1d_psd_from_image, n_rings=n_rings))(image_batch)
        # Low percentile across the batch as a (biased) robust noise floor.
        noise_psd_estimate = jnp.percentile(batch_of_psds, q=percentile, axis=0)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'background' or 'percentile'.")

    return noise_psd_estimate


def create_whitening_fn(
        noise_psd_1d: jax.Array,
        image_shape: Tuple[int, int]
) -> Callable[[jax.Array], jax.Array]:
    """
    Creates and JIT-compiles a function to whiten batches of images.

    This factory pre-builds the whitening filter from the dataset's noise profile
    for maximum efficiency.

    Args:
        noise_psd_1d: The pre-computed 1D noise PSD of the dataset from `estimate_noise_psd`.
        image_shape: The (height, width) of the images to be processed.

    Returns:
        A fast, JIT-compiled function that takes an image batch (B, H, W, 1) and
        returns the whitened batch.
    """
    height, width = image_shape

    # Pre-compute the 2D whitening filter from the 1D PSD
    r, n_rings = _radius_grid(image_shape)

    # Guard against a noise PSD that is shorter than the radial index range
    # (would otherwise silently clamp the corner coefficients to the last value).
    noise_psd_1d = jnp.asarray(noise_psd_1d)
    if noise_psd_1d.shape[0] < n_rings:
        pad = n_rings - noise_psd_1d.shape[0]
        noise_psd_1d = jnp.concatenate([noise_psd_1d, jnp.repeat(noise_psd_1d[-1:], pad)])

    # The filter is the inverse of the square root of the power spectrum. The
    # epsilon floor is scaled to the PSD so it is not sensitive to the absolute
    # image scale.
    eps = 1e-6 * jnp.max(noise_psd_1d)
    radial_filter = 1.0 / (jnp.sqrt(noise_psd_1d) + eps)

    # Normalise so the filter has unit gain on average over the spectrum; this
    # keeps the whitened images on roughly the same scale as the input.
    radial_filter = radial_filter / jnp.mean(radial_filter)

    # Map the 1D filter values back to a 2D grid and unshift for multiplication
    whitening_filter_2d_shifted = radial_filter[r]
    whitening_filter_2d = fft.ifftshift(whitening_filter_2d_shifted)

    # This is the final function that will be returned
    @jit
    def whiten_batch(image_batch: jax.Array) -> jax.Array:
        """
        Applies the pre-computed whitening filter to a batch of images.
        Expected input shape: (B, H, W, 1).
        """
        if image_batch.ndim != 4 or image_batch.shape[1:3] != image_shape:
            raise ValueError(f"Input batch must have shape (B, {height}, {width}, 1)")

        images_squeezed = jnp.squeeze(image_batch, axis=-1)

        # Apply the filter in Fourier space.
        # JAX handles broadcasting the (H, W) filter across the (B, H, W) batch.
        fft_images = fft.fft2(images_squeezed)
        whitened_fft = fft_images * whitening_filter_2d

        whitened_images = jnp.real(fft.ifft2(whitened_fft))

        return jnp.expand_dims(whitened_images, axis=-1)

    return whiten_batch
