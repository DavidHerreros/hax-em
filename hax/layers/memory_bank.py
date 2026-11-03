from flax import nnx
import jax
from jax import random as jnr
import jax.numpy as jnp


def sample_bank(bank, key, n_samples, n_valid=None):
    """Random subset of the rows of a memory bank that hold real data.

    The training programs use this to show the learnt landscape without running a
    second pass over the dataset: the bank already holds the latents the model has
    encoded during training. ``n_valid`` is how many rows have been enqueued so far
    (rows are filled from 0 upwards), so an early call does not sample the
    zero-initialised tail; pass ``None`` once the bank is known to be full.
    """
    n_rows = bank.shape[0]
    if n_valid is None:
        n_valid = n_rows
    n_valid = int(min(max(int(n_valid), 1), n_rows))
    n_samples = int(min(int(n_samples), n_valid))
    indices = jnr.choice(key, a=jnp.arange(n_valid), shape=(n_samples,), replace=False)
    return bank[indices]


class MemoryBank(nnx.Module):
    def __init__(self, buffer_size=None, n_dim=None, array_init=None):
        super(MemoryBank, self).__init__()
        self.buffer_size = buffer_size

        if array_init is None and buffer_size is not None and n_dim is not None:
            array_init = jnp.zeros((buffer_size, n_dim))
        elif array_init is not None:
            self.buffer_size = array_init.shape[0]
        else:
            raise ValueError("Provide either array_init or buffer_size/n_dim parameters")

        self.memory_bank = nnx.Variable(array_init)
        self.memory_bank_ptr = nnx.Variable(jnp.zeros((1,), dtype=jnp.int32))

    # --- Method for enqueuing to the memory bank ---
    def enqueue(self, keys_to_add):
        """Updates the memory bank and pointer using JIT-compatible operations."""
        ptr = self.memory_bank_ptr.get_value()[0]
        batch_size = keys_to_add.shape[0]

        indices = (jnp.arange(batch_size) + ptr) % self.buffer_size
        self.memory_bank.value = self.memory_bank.get_value().at[indices].set(keys_to_add)

        self.memory_bank_ptr.value = jnp.array(
            [(ptr + batch_size) % self.buffer_size], dtype=jnp.int32
        )

        # # Define the starting position for the update.
        # # It must be a tuple with one index per dimension of the array.
        # # Our memory_bank is 2D, so we need (start_row, start_column).
        # start_indices = (ptr, 0)
        #
        # # Use `lax.dynamic_update_slice` instead of `.at[...].set(...)`
        # self.memory_bank.value = jax.lax.dynamic_update_slice(
        #     self.memory_bank.get_value(), # 1. The original large array to be updated
        #     keys_to_add,                  # 2. The smaller array containing the new data
        #     start_indices                 # 3. The dynamic starting position
        # )
        #
        # # The pointer update logic remains the same, as it's just arithmetic
        # current_batch_size = keys_to_add.shape[0]
        # self.memory_bank_ptr.value = jnp.array(
        #     [(ptr + current_batch_size) % self.buffer_size]
        # )

    def get(self):
        return self.memory_bank.get_value()
