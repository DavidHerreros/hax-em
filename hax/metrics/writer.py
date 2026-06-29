#!/usr/bin/env python


import os
import numpy as np
from datetime import datetime

from tensorboardX import SummaryWriter

import jax
import jax.numpy as jnp
from jax import device_get
from jax.numpy import ndarray as JaxArray

from hax.utils import min_max_scale, low_pass_3d


class JaxSummaryWriter(SummaryWriter):
    def __init__(self, log_dir=None, **kwargs):
        super(JaxSummaryWriter, self).__init__(log_dir=os.path.join(log_dir, datetime.now().strftime("%Y%m%d-%H%M%S")), **kwargs)

    def _to_numpy(self, x):
        # If it’s a JAX array, pull it to host and convert
        if isinstance(x, JaxArray):
            return np.asarray(device_get(x))
        return x

    def _convert_tree(self, tree):
        # Recursively map over lists/tuples/dicts
        return jax.tree_util.tree_map(self._to_numpy, tree)

    def add_volumes_slices(self, volumes):
        @jax.jit
        def prepare_slices(volumes):
            # 0) From gray image to color
            def gray_to_color(x):
                x_flat = x.reshape(-1)

                # Robust normalization to [0, 1]; if x is constant, map to mid (0.5 → white)
                x_min = jnp.min(x_flat)
                x_max = jnp.max(x_flat)
                denom = x_max - x_min
                t = jnp.where(denom > 0, (x_flat - x_min) / denom, jnp.full_like(x_flat, 0.5))

                # bwr: blue (0,0,1) -> white (1,1,1) over [0,t_limit], then white -> red (1,0,0) over (t_limit,t_max]
                blue = jnp.array([0.0, 0.0, 1.0])
                white = jnp.array([1.0, 1.0, 1.0])
                red = jnp.array([1.0, 0.0, 0.0])

                # Interp weights for each half
                t_limit = 0.25
                t_max = 1.0
                t_lo = jnp.clip(t / t_limit, 0.0, t_max)  # [0,t_limit] segment
                t_hi = jnp.clip((t - t_limit) / t_limit, 0.0, t_max)  # (t_limit,t_max] segment

                # Colors for each segment
                color_lo = (1.0 - t_lo[:, None]) * blue + t_lo[:, None] * white
                color_hi = (1.0 - t_hi[:, None]) * white + t_hi[:, None] * red

                # Piecewise select
                color = jnp.where((t[:, None] <= t_limit), color_lo, color_hi)

                return color.reshape(x.shape + (3,))

            # 1) Compute MAD and mean volume
            volume_mad = volumes.std(axis=0)
            volume_mean = volumes.mean(axis=0)

            # 2) Filter volumes
            volume_mad = low_pass_3d(volume_mad, std=1.)
            volume_mean = low_pass_3d(volume_mean, std=1.)

            # 3) Convert volumes to color
            volume_mean_color = min_max_scale(volume_mean)[..., None]
            volume_mad_color = gray_to_color(min_max_scale(volume_mad))

            # 4) Extract central slices
            central_slice = int(0.5 * volume_mean.shape[-1])
            slices_mean = [volume_mean_color[central_slice, :, :, :], volume_mean_color[:, central_slice, :, :], volume_mean_color[:, :, central_slice, :]]  # (Z, Y, X)
            slices_mad = [volume_mad_color[central_slice, :, :, :], volume_mad_color[:, central_slice, :, :], volume_mad_color[:, :, central_slice, :]]  # (Z, Y, X)
            return jnp.stack(slices_mean, axis=0), jnp.stack(slices_mad, axis=0)

        # 1) Prepare slices from volumes
        slices_mean, slices_mad = prepare_slices(volumes)

        # 2) Log images in Tensorboard
        self.add_image("Consensus volume", slices_mean, dataformats="NHWC")
        self.add_image("Variation volume", slices_mad, dataformats="NHWC")

        # 3) Add text to explain colors in the visualizations
        legend_mean = """
        <h3>Consensus volume color legend</h3>
        <ul>
            <li>Consensus volume predicted by the network — White surface</li>
        </ul>
        """
        legend_mad = """
        <h3>Variation volume color legend</h3>
        <ul>
            <li>No structural variation — Blue surface</li>
            <li>Structural variation — White (medium)/Red (maximum) surface</li>
        </ul>
        """
        self.add_text("Consensus volume color legend", legend_mean)
        self.add_text("Variation volume color legend", legend_mad)

    def __getattribute__(self, name):
        # 1) Always let internal/private names through unwrapped:
        if name.startswith("_"):
            return object.__getattribute__(self, name)

        # 2) If the base class doesn’t define this attr, pass it straight through:
        if not hasattr(SummaryWriter, name):
            return object.__getattribute__(self, name)

        # 3) Grab the real method from *self* (so bound correctly):
        target = object.__getattribute__(self, name)

        # 4) If it isn’t callable, return it as-is:
        if not callable(target):
            return target

        # 5) Otherwise return our wrapper:
        def wrapped(*args, **kwargs):
            # cheap check: do we actually need to convert?
            leaves = jax.tree_util.tree_leaves((args, kwargs))
            if not any(isinstance(x, JaxArray) for x in leaves):
                return target(*args, **kwargs)

            # only pay conversion cost when we see a JAX array
            args_np = self._convert_tree(args)
            kwargs_np = self._convert_tree(kwargs)
            return target(*args_np, **kwargs_np)

        return wrapped




def main():
    import argparse
    import time
    from tensorboard import program
    from hax.utils import bcolors

    parser = argparse.ArgumentParser(
        description="Launch TensorBoard for a given JAX SummaryWriter log directory"
    )
    parser.add_argument(
        "--logdir", type=str, required=True,
        help="Path to the TensorBoard log directory"
    )
    args = parser.parse_args()

    # Launch TensorBoard programmatically
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', args.logdir])
    url = tb.launch()
    print(f"{bcolors.WARNING}TensorBoard is running at {url}")
    print(f"Press {bcolors.UNDERLINE}{bcolors.BOLD}Ctrl+C{bcolors.ENDC} {bcolors.WARNING}to stop.{bcolors.ENDC}")

    # Block until Ctrl-C
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nReceived Ctrl+C—shutting down.")

if __name__ == "__main__":
    main()
