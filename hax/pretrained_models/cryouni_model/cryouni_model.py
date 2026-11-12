import os
import functools

import jax
import jax.numpy as jnp
from flax import nnx

import torch

from cryouni.coach_pl.configuration import CfgNode
from cryouni.coach_pl.model import build_model

import torchax
from torchax import interop

from hax.utils import prepare_image_cryocrab


class CryoUni:
    def __init__(self, input_shape=None):
        self.input_shape = input_shape

        # Load Config
        import hax.pretrained_models.cryouni_model as cryouni_module
        cfg_yaml = os.path.join(os.path.dirname(cryouni_module.__file__), "custom.yaml")
        cfg = CfgNode.load_yaml_with_base(cfg_yaml)
        cfg.DATAMODULE.DATASET.IMAGE_SIZE.SPATIAL = input_shape
        cfg.DATAMODULE.DATASET.IMAGE_SIZE.HARTLEY = input_shape
        cfg.MODEL.BACKBONE.PRETRAINED_PATH = None
        CfgNode.set_readonly(cfg, True)

        # Load model
        weights_path = os.path.join(os.path.dirname(cryouni_module.__file__), "model_weights", "cryouni-b.ckpt")
        model = build_model(cfg)
        model.load_pretrained(weights_path)
        model.eval()

        with torchax.default_env():
            model.to("jax")
            self.model = interop.JittableModule(model, extra_jit_args={"static_argnames": ("forward_head",)})

    def __call__(self, x, ctf=None):
        if ctf is not None:
            x = prepare_image_cryocrab(x, ctf)

        if x.shape[-1] == 1:
            x = x[..., 0]

        if self.input_shape is not None:
            x = jax.image.resize(x, (x.shape[0], self.input_shape, self.input_shape), method="bilinear")

        with torchax.default_env():
            pred = self.model(y_real=interop.torch_view(x), forward_head=False)
            pred = {key: interop.jax_view(value) for key, value in pred.items()}
        return pred


class CryoUniNNX(nnx.Module):
    def __init__(self, input_shape=None):
        self.input_shape = input_shape

        # Load Config
        import hax.pretrained_models.cryouni_model as cryouni_module
        cfg_yaml = os.path.join(os.path.dirname(cryouni_module.__file__), "custom.yaml")
        cfg = CfgNode.load_yaml_with_base(cfg_yaml)
        cfg.MODEL.BACKBONE.PRETRAINED_PATH = None
        cfg.DATAMODULE.DATASET.IMAGE_SIZE.SPATIAL = input_shape
        cfg.DATAMODULE.DATASET.IMAGE_SIZE.HARTLEY = input_shape
        CfgNode.set_readonly(cfg, True)

        # Load model
        weights_path = os.path.join(os.path.dirname(cryouni_module.__file__), "model_weights", "cryouni-b.ckpt")
        model = build_model(cfg)
        model.load_pretrained(weights_path)
        model.eval()

        with torchax.default_env():
            model.to("jax")
            self.model = interop.JittableModule(model, extra_jit_args={"static_argnames": ("forward_head",)})

            self.model_fn = functools.partial(self.model.functional_call, "forward")

            self.params = nnx.Dict({
                k: nnx.Param(interop.jax_view(v))
                for k, v in self.model.params.items()
            })

            self.buffers = nnx.Dict({
                k: interop.jax_view(v)
                for k, v in self.model.buffers.items()
            })

    def unwrap_nnx_value(self, x):
        return x.get_value() if hasattr(x, "value") else x

    def to_torchax_tensor(self, x):
        return interop.torch_view(self.unwrap_nnx_value(x))

    def __call__(self, x, ctf=None):
        if ctf is not None:
            x = prepare_image_cryocrab(x, ctf)

        if x.shape[-1] == 1:
            x = x[..., 0]

        if self.input_shape is not None:
            x = jax.image.resize(x, (x.shape[0], self.input_shape, self.input_shape), method="bilinear")

        with torchax.default_env():
            params = {k: self.to_torchax_tensor(v) for k, v in self.params.items()}
            buffers = {k: self.to_torchax_tensor(v) for k, v in self.buffers.items()}

            pred = torch.func.functional_call(self.model, {**params, **buffers}, args=(x,))

            pred = {key: interop.jax_view(value) for key, value in pred.items()}
        return pred



class LayerNorm2dNHWC(nnx.Module):
    """
    Equivalent to applying LayerNorm over the channel dimension
    for each spatial position.

    Input shape:
        (B, H, W, C)
    Output shape:
        (B, H, W, C)
    """

    def __init__(
        self,
        num_features: int,
        *,
        epsilon: float = 1e-6,
        rngs: nnx.Rngs,
    ):
        self.norm = nnx.LayerNorm(
            num_features=num_features,
            epsilon=epsilon,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.norm(x)


class Identity(nnx.Module):
    def __call__(self, x):
        return x


class CryoUniHead(nnx.Module):
    def __init__(
        self,
        image_size: int,
        embed_channels: int = 2,
        in_channels: int = 768,
        cls_norm: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        patch_size = 16
        self.in_channels = in_channels
        self.embed_channels = embed_channels
        self.grid_size = image_size // patch_size
        self.out_shape = in_channels + (image_size // patch_size) * (image_size // patch_size) * embed_channels

        self.neck_patch_norm = LayerNorm2dNHWC(
            in_channels,
            rngs=rngs,
        )

        self.neck_patch_conv = nnx.Conv(
            in_features=in_channels,
            out_features=embed_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            rngs=rngs,
        )

        if cls_norm:
            self.neck_cls = nnx.LayerNorm(
                num_features=in_channels,
                rngs=rngs,
            )
        else:
            self.neck_cls = Identity()

    def __call__(
        self,
        cls_tokens: jax.Array,
        patch_tokens: jax.Array,
    ) -> jax.Array:
        """
        Args:
            cls_tokens:
                Shape (B, in_channels)

            patch_tokens:
                Shape (B, L, in_channels), where
                L = grid_size * grid_size

        Returns:
            tokens:
                Concatenated features from class and patch tokens
        """

        B, L, E = patch_tokens.shape

        expected_L = self.grid_size * self.grid_size
        if L != expected_L:
            raise ValueError(
                f"Expected {expected_L} patch tokens, got {L}."
            )

        if E != self.in_channels:
            raise ValueError(
                f"Expected patch token dim {self.in_channels}, got {E}."
            )

        # PyTorch:
        #   patch_tokens: (B, L, E)
        #   -> (B, H, W, E)
        #
        # In JAX/Flax we keep NHWC for convolution.
        patch_tokens = patch_tokens.reshape(
            B,
            self.grid_size,
            self.grid_size,
            E,
        )

        # LayerNorm2d + Conv2d equivalent in NHWC.
        patch_tokens = self.neck_patch_norm(patch_tokens)
        patch_tokens = self.neck_patch_conv(patch_tokens)

        # Current shape:
        #   (B, H, W, embed_channels)
        #
        # PyTorch flattened after converting to NCHW:
        #   (B, embed_channels, H, W) -> (B, embed_channels * H * W)
        #
        # To preserve the same flattening order, transpose to NCHW first.
        patch_tokens = jnp.transpose(patch_tokens, (0, 3, 1, 2))
        patch_tokens = patch_tokens.reshape(B, -1)

        # Optional class-token LayerNorm.
        cls_tokens = self.neck_cls(cls_tokens)

        # Concatenate:
        #   cls_tokens:   (B, in_channels)
        #   patch_tokens: (B, embed_channels * H * W)
        #
        # Result:
        #   tokens:       (B, in_channels + embed_channels * H * W)
        tokens = jnp.concatenate([cls_tokens, patch_tokens], axis=1)

        return tokens

