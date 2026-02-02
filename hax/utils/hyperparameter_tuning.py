import jax
import jax.numpy as jnp
from hax.utils.loggers import bcolors


def find_max_batch_size(graphdef, state, step, md, rng, input_shape_per_sample):
    """
    Empirically finds the largest power-of-two batch size that fits in GPU memory.
    """

    batch_size = 8
    max_fitted_batch_size = 0

    print("Testing batch sizes...")

    while True:
        try:
            # Create dummy batch
            x_shape = (batch_size,) + input_shape_per_sample

            x_dummy = jnp.ones(x_shape)
            labels_dummy = jnp.zeros((batch_size,), dtype=jnp.int32)

            # Force compilation and execution
            # We use block_until_ready() to ensure the async dispatch actually finishes
            _ = step(graphdef, state, x_dummy, labels_dummy, md, rng, do_update=False)
            # loss.block_until_ready()

            print(f"  Batch size {batch_size}: {bcolors.OKGREEN}OK{bcolors.ENDC}")
            max_fitted_batch_size = batch_size

            # Double the batch size
            batch_size *= 2

            # Optional: Break if batch size is unreasonably large (above 256 weird errors may appear)
            if batch_size > 256:
                break

        except (RuntimeError, jax.errors.JaxRuntimeError) as e:
            # Catch OOM errors. Different setups might raise slightly different errors,
            # but usually they contain "Resource exhausted" or "Out of memory"
            if "Resource exhausted" in str(e) or "Out of memory" in str(e):
                print(f"  Batch size {batch_size}: OOM (Limit reached)")
                break
            else:
                raise e  # Re-raise if it's a different error

        # Clear backend caches to free up fragmented memory between attempts
        jax.clear_caches()

    return max_fitted_batch_size // 2