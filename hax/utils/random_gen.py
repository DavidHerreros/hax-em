import jax.numpy as jnp
import jax.random as jnr

def random_rotation_matrices(batch_size, key):
    # Generate batch_size random numbers for u1, u2, and u3. The three draws
    # must come from independent subkeys: reusing one key returns the same
    # values three times, which collapses the samples onto a biased
    # one-parameter family instead of covering SO(3).
    key1, key2, key3 = jnr.split(key, 3)
    u1 = jnr.uniform(key1, shape=(batch_size,), minval=0.0, maxval=1.0)
    u2 = jnr.uniform(key2, shape=(batch_size,), minval=0.0, maxval=1.0)
    u3 = jnr.uniform(key3, shape=(batch_size,), minval=0.0, maxval=1.0)

    # Compute quaternion components using the method from Shoemake (1992).
    q1 = jnp.sqrt(1. - u1) * jnp.sin(2. * jnp.pi * u2)
    q2 = jnp.sqrt(1. - u1) * jnp.cos(2. * jnp.pi * u2)
    q3 = jnp.sqrt(u1)      * jnp.sin(2. * jnp.pi * u3)
    q4 = jnp.sqrt(u1)      * jnp.cos(2. * jnp.pi * u3)

    # Use the convention (w, x, y, z) = (q4, q1, q2, q3).
    w, x, y, z = q4, q1, q2, q3

    # Compute the elements of the rotation matrix.
    r00 = 1. - 2. * (y * y + z * z)
    r01 = 2. * (x * y - z * w)
    r02 = 2. * (x * z + y * w)

    r10 = 2. * (x * y + z * w)
    r11 = 1. - 2. * (x * x + z * z)
    r12 = 2. * (y * z - x * w)

    r20 = 2. * (x * z - y * w)
    r21 = 2. * (y * z + x * w)
    r22 = 1. - 2. * (x * x + y * y)

    # Each of these tensors has shape (batch_size,).
    # Now, form the rotation matrices for each batch element.
    row0 = jnp.stack([r00, r01, r02], axis=1)  # shape: (batch_size, 3)
    row1 = jnp.stack([r10, r11, r12], axis=1)  # shape: (batch_size, 3)
    row2 = jnp.stack([r20, r21, r22], axis=1)  # shape: (batch_size, 3)

    # Stack the rows to form a (batch_size, 3, 3) tensor.
    R = jnp.stack([row0, row1, row2], axis=1)
    return R
