#!/usr/bin/env python


import jax
from flax import nnx
from hax.utils import *


def main():
    import os
    import inspect
    import numpy as np
    import argparse
    from xmipp_metadata.image_handler import ImageHandler
    from hax.checkpointer import NeuralNetworkCheckpointer
    from hax.cli import common_args as ca

    parser = argparse.ArgumentParser()
    parser.add_argument("--latents_file", required=True, type=str,
                        help="File with the latent vectors needed to decode the volumes. "
                             f"Valid extensions include {bcolors.ITALIC}.txt{bcolors.ENDC} and {bcolors.ITALIC}.npy{bcolors.ENDC}")
    parser.add_argument("--reload", required=True, type=str,
                        help=f"Path to a folder containing an already saved neural network ({bcolors.WARNING}NOTE{bcolors.ENDC}: "
                             f"Only networks saved in Pickled format can be supplied here.")
    parser.add_argument("--output_path", required=True, type=str,
                        help=f"Path were the decoded volumes will be saved.")
    parser.add_argument("--decode_motion", action=argparse.BooleanOptionalAction, default=True,
                        help=f"Include the {bcolors.ITALIC}deformation field{bcolors.ENDC} (mass transport) when decoding. "
                             f"With {bcolors.ITALIC}--no-decode_motion{bcolors.ENDC} the Gaussians are held at their consensus "
                             f"positions, so the decoded volume shows only what the {bcolors.ITALIC}occupancy{bcolors.ENDC} "
                             f"pathway contributes. Combined with {bcolors.ITALIC}--decode_occupancy{bcolors.ENDC} this gives the "
                             f"four ablations (both / motion only / occupancy only / consensus) needed to tell a genuine domain "
                             f"motion from an amplitude change: a real motion survives {bcolors.ITALIC}--no-decode_occupancy"
                             f"{bcolors.ENDC}, real compositional change survives {bcolors.ITALIC}--no-decode_motion{bcolors.ENDC}. "
                             f"Only meaningful for networks trained with mass transport.")
    parser.add_argument("--decode_occupancy", action=argparse.BooleanOptionalAction, default=True,
                        help=f"Include the per-Gaussian {bcolors.ITALIC}occupancy{bcolors.ENDC} (amplitude) change when decoding. "
                             f"With {bcolors.ITALIC}--no-decode_occupancy{bcolors.ENDC} the amplitudes are pinned to the consensus, "
                             f"so the decoded volume is the deformation field acting {bcolors.ITALIC}alone{bcolors.ENDC} and is "
                             f"exactly mass-conserving -- any density that moves between two sites is genuine transport, not a "
                             f"fade-out/fade-in. See {bcolors.ITALIC}--decode_motion{bcolors.ENDC}.")
    parser.add_argument("--export_deformation", action=argparse.BooleanOptionalAction, default=True,
                        help=f"Also export the decoder's per-state {bcolors.ITALIC}deformation field{bcolors.ENDC} "
                             f"(the deformed point cloud + per-point occupancy relative to the consensus) "
                             f"under {bcolors.UNDERLINE}<output_path>/deformation/{bcolors.ENDC}. This is the exact "
                             f"correspondence data downstream tools (e.g. the hax-em assistant's regional-motion "
                             f"analysis) use to characterise domain motions — hinges, rotations, coordinated motion "
                             f"and compositional/occupancy change — rigorously rather than guessing from the maps. "
                             f"Best-effort: skipped with a warning if the loaded network exposes no deformation field. "
                             f"Use {bcolors.ITALIC}--no-export_deformation{bcolors.ENDC} to disable.")
    args = ca.parse_with_config(parser)

    # Read latent vectors
    if args.latents_file.endswith(".txt"):
        latents = np.loadtxt(args.latents_file)
    elif args.latents_file.endswith(".npy"):
        latents = np.load(args.latents_file)
    else:
        raise ValueError(f"The format of {bcolors.ITALIC}latents_file{bcolors.ENDC} is not valid. Pease, provide a file "
                         f"saved as {bcolors.ITALIC}.txt{bcolors.ENDC} or {bcolors.ITALIC}.npy{bcolors.ENDC}")

    # Reload neural network
    network = NeuralNetworkCheckpointer.load(args.reload)

    # Split network (faster JIT)
    graphdef, state = nnx.split(network)

    # Jit function to increase speed
    ablate = (not args.decode_motion) or (not args.decode_occupancy)
    supports_ablation = "motion" in inspect.signature(type(network).decode_volume).parameters
    if supports_ablation:
        supports_ablation = getattr(getattr(network, "delta_volume_decoder", None),
                                    "transport_mass", False)

    if ablate and not supports_ablation:
        raise SystemExit(
            f"{bcolors.FAIL}--no-decode_motion / --no-decode_occupancy are only supported for networks "
            f"trained with mass transport (HetSIREN with {bcolors.ITALIC}--transport_mass{bcolors.ENDC}"
            f"{bcolors.FAIL}). The network loaded from '{args.reload}' ({type(network).__name__}) has no "
            f"separable motion/occupancy pathways, so there is nothing to ablate.{bcolors.ENDC}")

    @jax.jit
    def decode_volume(graphdef, state, x):
        model = nnx.merge(graphdef, state)
        if ablate:
            return model.decode_volume(x, motion=args.decode_motion, occupancy=args.decode_occupancy)
        return model.decode_volume(x)

    # Decode and save the volumes
    volumes = []
    for latent in latents:
        volumes.append(np.array(decode_volume(graphdef, state, latent)))

    # Save volumes to file
    for idx in range(latents.shape[0]):
        ImageHandler().write(volumes[idx], os.path.join(args.output_path, f"decoded_volume_{idx:04d}.mrc"), overwrite=True)

    # Optionally export the deformation field (exact correspondences) so downstream
    # tools can characterise domain motions rigorously instead of inferring them
    # from the rasterised maps
    if getattr(args, "export_deformation", False):
        _export_deformation(network, graphdef, state, latents, args.output_path, np)


def _export_deformation(network, graphdef, state, latents, output_path, np):
    """Write <output_path>/deformation/{rest_points,deformed_points,occupancy}.npy
    plus meta.json, using the model's common ``decode_field`` accessor. Occupancy is
    per-state when the decoder exposes it (composition/occupancy change), else the
    static consensus values. Any failure downgrades to a warning."""
    import os
    import json
    from flax import nnx

    if not hasattr(network, "decode_field"):
        print(f"{bcolors.WARNING}--export_deformation:{bcolors.ENDC} the loaded network exposes no "
              f"deformation field (decode_field); skipping deformation export.")
        return

    has_dvd = hasattr(network, "delta_volume_decoder")           # per-state occupancy
    static_values = None
    if not has_dvd and hasattr(network, "values"):
        static_values = np.asarray(network.values)               # Zernike: fixed occupancy

    @jax.jit
    def decode_field(graphdef, state, x):
        model = nnx.merge(graphdef, state)
        return model.decode_field(x[None, ...])

    @jax.jit
    def decode_occupancy(graphdef, state, x):
        model = nnx.merge(graphdef, state)
        _, values = model.delta_volume_decoder(x[None, ...])
        return values

    try:
        rest = None
        deformed, occ = [], []
        for latent in latents:
            field, rest_c = decode_field(graphdef, state, latent)
            field = np.array(field)[0]                            # (N, 3), box-fraction
            rest = np.array(rest_c)[0] if rest is None else rest  # (N, 3), same every state
            deformed.append(rest + field)
            if has_dvd:
                occ.append(np.array(decode_occupancy(graphdef, state, latent))[0].reshape(-1))
            elif static_values is not None:
                occ.append(static_values.reshape(-1))

        outdir = os.path.join(output_path, "deformation")
        os.makedirs(outdir, exist_ok=True)
        np.save(os.path.join(outdir, "rest_points.npy"), rest.astype(np.float32))
        np.save(os.path.join(outdir, "deformed_points.npy"),
                np.stack(deformed).astype(np.float32))            # (S, N, 3)
        if occ:
            np.save(os.path.join(outdir, "occupancy.npy"),
                    np.stack(occ).astype(np.float32))             # (S, N)
        meta = {
            "n_states": int(len(deformed)),
            "n_points": int(rest.shape[0]),
            "decoder": type(network).__name__,
            "occupancy_per_state": bool(has_dvd),
            "xsize": int(getattr(network, "xsize", 0)) or None,
            "sampling_rate": float(getattr(network, "sr", 0.0)) or None,
            "frame": "box-fraction (multiply by 0.5*xsize for voxels, then by sampling_rate for Angstrom)",
        }
        with open(os.path.join(outdir, "meta.json"), "w") as fh:
            json.dump(meta, fh, indent=2)
        print(f"{bcolors.OKGREEN}--export_deformation:{bcolors.ENDC} wrote deformation field for "
              f"{meta['n_states']} states ({meta['n_points']} points) to {outdir}")
    except Exception as exc:  # noqa: BLE001 - never break the (already written) volumes
        print(f"{bcolors.WARNING}--export_deformation:{bcolors.ENDC} could not export the deformation "
              f"field ({exc}); volumes were written normally.")
