from jax import numpy as jnp


def min_max_scale(image):
    min_val = jnp.min(image)
    max_val = jnp.max(image)
    # Add a small epsilon to prevent division by zero if the image is flat
    return (image - min_val) / (max_val - min_val + 1e-6)


def standard_normalization(images, mask=None):
    axis = (1, 2) if images.ndim == 3 else (1, 2, 3)

    if mask is None:
        images = images - jnp.mean(images, axis=axis, keepdims=True)
        return images / (jnp.std(images, axis=axis, keepdims=True) + 1e-8)

    m = jnp.broadcast_to(mask.reshape((1,) + mask.shape + (1,) * (images.ndim - 3)), images.shape)
    n = jnp.maximum(jnp.sum(m, axis=axis, keepdims=True), 1.0)
    mean = jnp.sum(images * m, axis=axis, keepdims=True) / n
    var = jnp.sum(jnp.square(images - mean) * m, axis=axis, keepdims=True) / n
    return (images - mean) / (jnp.sqrt(var) + 1e-8) * m


def logistic_transform_std_shift(self, errors, mu=None, sigma=None):
    """
    Transforms a 1D array of errors to a 0-1 scale using a logistic function,
    such that a value of 0.5 corresponds to errors that are +1 standard deviation away from the mean.

    The transformation is defined as:
        f(x) = 1 / (1 + exp(-k*(x - (mu+sigma))))
    with k chosen so that:
        f(mu)   ~ 0.25  and  f(mu+2*sigma) ~ 0.75.

    Parameters:
        errors (np.array): 1D array of error values.

    Returns:
        np.array: Transformed error values in the interval (0, 1).
    """
    # Compute the mean and standard deviation
    mu = jnp.mean(errors) if mu is None else mu
    sigma = jnp.std(errors) if sigma is None else sigma

    # Prevent division by zero in case sigma is zero
    if sigma == 0:
        return jnp.full_like(errors, 0.5)

    # Choose k such that one std below and above the center give 0.25 and 0.75 respectively.
    k = jnp.log(3) / sigma

    # Shift the logistic function center to mu + sigma so that f(mu+sigma) = 0.5
    transformed = 1 / (1 + jnp.exp(-k * (errors - (mu + sigma))))
    return transformed
