import jax
import jax.numpy as jnp


def normal_initializer_mean(stddev=1e-2, mean=0.0):
  """
  Creates a JAX normal initializer with a custom mean and standard deviation.

  Args:
    stddev: The standard deviation of the normal distribution.
    mean: The mean of the normal distribution.

  Returns:
    A JAX initializer function.
  """
  def init(key, shape, dtype=jnp.float_):
    return jax.random.normal(key, shape, dtype) * stddev + mean
  return init


def uniform(minval=-1, maxval=1):
  def init(key, shape, dtype=jnp.float_):
    return jax.random.uniform(key, shape, dtype, minval=minval, maxval=maxval)
  return init


def rot6d_perturbation_init(N: int, sigma: float = 0.01, mode: str = ""):
  sigma = sigma * jnp.pi / 180.

  if mode == "bias":
    """For the BIAS of the final linear layer, shape (N * 6,)."""
    def init(key, shape, dtype=jnp.float32):
      assert shape == (N * 6,), f"Expected shape ({N * 6},), got {shape}"
      keys = jax.random.split(key, N)

      def single_rot6d(key_i):
        omega = jax.random.normal(key_i, (3,)) * sigma
        angle = jnp.linalg.norm(omega)
        wx, wy, wz = omega
        W = jnp.array([[0, -wz, wy],
                       [wz, 0, -wx],
                       [-wy, wx, 0]])
        delta_R = (jnp.eye(3)
                   + jnp.sinc(angle / jnp.pi) * jnp.pi * W
                   + (1 - jnp.cos(angle)) / (angle ** 2 + 1e-8) * (W @ W))
        return delta_R[:, :2].T.ravel()

      return jax.vmap(single_rot6d)(keys).ravel().astype(dtype)  # (N * 6,)

  elif mode == "weight":
    """For the WEIGHT MATRIX, shape (input_dim, N * 6)."""
    def init(key, shape, dtype=jnp.float32):
      assert shape[-1] == N * 6, f"Expected last dim {N * 6}, got {shape[-1]}"
      input_dim = shape[0]
      total = input_dim * N
      keys = jax.random.split(key, total)  # (input_dim * N, 2)

      def single_rot6d(key_i):
        omega = jax.random.normal(key_i, (3,)) * sigma
        angle = jnp.linalg.norm(omega)
        wx, wy, wz = omega
        W = jnp.array([[0, -wz, wy],
                       [wz, 0, -wx],
                       [-wy, wx, 0]])
        delta_R = (jnp.eye(3)
                   + jnp.sinc(angle / jnp.pi) * jnp.pi * W
                   + (1 - jnp.cos(angle)) / (angle ** 2 + 1e-8) * (W @ W))
        return delta_R[:, :2].T.ravel()

      rot6ds = jax.vmap(single_rot6d)(keys)  # (input_dim * N, 6)
      return rot6ds.reshape(input_dim, N * 6).astype(dtype)

  else:
    raise ValueError("Parameter mode must be 'bias' or 'weight'")

  return init
