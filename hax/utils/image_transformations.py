import jax
import jax.numpy as jnp

from .fourier_filters import ctfFilter, wiener2DFilter


def apply_batch_translations(images, translations):
    """
    Applies (B, 2) translations to (B, M, M, 1) images.

    Args:
        images: Array of shape (B, M, M, 1)
        translations: Array of shape (B, 2) -> [dy, dx]
    """
    B, M, _, C = images.shape

    # Define the operation for a single image
    def shift_single(img, translation):
        return jax.image.scale_and_translate(
            image=img,
            shape=img.shape,  # Output shape (M, M, 1)
            spatial_dims=(0, 1),  # Only scale/translate the M x M dims
            scale=jnp.ones(2),  # No scaling (1.0 for all dims)
            translation=translation,
            method='linear'  # Can also use 'cubic' or 'lanczos3'
        )

    # Vectorize over the batch dimension
    translations = jnp.stack([translations[..., 1], translations[..., 0]], axis=-1)  # [dy, dx]
    return jax.vmap(shift_single)(images, translations)


def prepare_image_cryocrab(x, ctf):
    # Phase flip image
    ctf_mask = jnp.where(ctf < 0, -1.0, 1.0)
    x = ctfFilter(x[..., 0], ctf_mask, pad_factor=2)[..., None]

    # Constrast normalization
    # min_val = jnp.min(x, axis=(1, 2), keepdims=True)
    # max_val = jnp.max(x, axis=(1, 2), keepdims=True)
    # x = (x - min_val) / (max_val - min_val + 1e-8)

    # Contrast normalization (robust)
    lo, hi = jnp.percentile(x, jnp.array([0.5, 99.5]))
    denom = hi - lo
    clipped = jnp.clip(x, lo, hi)
    normalized = 2.0 * (clipped - lo) / (denom + 1e-8) - 1.0
    x = jnp.where(denom < 1e-8, x, normalized).astype(jnp.float32)

    # Z-Score standarization
    mean_val = jnp.mean(x, axis=(1, 2), keepdims=True)
    std_val = jnp.std(x, axis=(1, 2), keepdims=True)
    x = (x - mean_val) / (std_val + 1e-8)

    return x

def prepare_image_wiener(x, ctf):
    # Apply wiener filter
    x = wiener2DFilter(x[..., 0], ctf, pad_factor=2)[..., None]


    # Constrast normalization
    # min_val = jnp.min(x, axis=(1, 2), keepdims=True)
    # max_val = jnp.max(x, axis=(1, 2), keepdims=True)
    # x = (x - min_val) / (max_val - min_val + 1e-8)

    # Contrast normalization (robust)
    lo, hi = jnp.percentile(x, jnp.array([0.5, 99.5]))
    denom = hi - lo
    clipped = jnp.clip(x, lo, hi)
    normalized = 2.0 * (clipped - lo) / (denom + 1e-8) - 1.0
    x = jnp.where(denom < 1e-8, x, normalized).astype(jnp.float32)

    # Z-Score standarization
    mean_val = jnp.mean(x, axis=(1, 2), keepdims=True)
    std_val = jnp.std(x, axis=(1, 2), keepdims=True)
    x = (x - mean_val) / (std_val + 1e-8)

    return x