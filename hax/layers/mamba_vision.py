import jax
from flax import nnx


class MambaBlock(nnx.Module):
    """A simplified Selective SSM (Mamba) block for visual encoding."""
    def __init__(self, dim: int, rngs: nnx.Rngs):
        self.norm = nnx.LayerNorm(dim, rngs=rngs)
        # Selective scan parameters (simplified for architecture demo)
        self.proj_in = nnx.Linear(dim, dim * 2, rngs=rngs)
        self.conv1d = nnx.Conv(dim * 2, dim * 2, kernel_size=(3,), rngs=rngs)
        self.proj_out = nnx.Linear(dim * 2, dim, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        res = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = nnx.silu(self.conv1d(x)) # Local temporal/spatial dependency
        x = self.proj_out(x)
        return x + res