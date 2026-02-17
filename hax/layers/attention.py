import jax.numpy as jnp
from flax import nnx
from einops import rearrange


class Attention(nnx.Module):
    def __init__(self, C: int, num_heads: int, dropout_prob: float, *, rngs: nnx.Rngs):
        self.proj1 = nnx.Linear(C, C * 3, rngs=rngs, dtype=jnp.bfloat16)
        self.proj2 = nnx.Linear(C, C, rngs=rngs, dtype=jnp.bfloat16)
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob

    def __call__(self, x, train):
        h, w = x.shape[1], x.shape[2]
        x = rearrange(x, 'b h w c -> b (h w) c')
        x = self.proj1(x)
        x = rearrange(x, 'b L (C H K) -> K b H L C', K=3, H=self.num_heads)
        q, k, v = x[0], x[1], x[2]
        x = nnx.dot_product_attention(q, k, v)
        x = rearrange(x, 'b H (h w) C -> b h w (H C)', h=h, w=w)
        x = self.proj2(x)
        return x