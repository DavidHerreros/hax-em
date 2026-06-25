#!/usr/bin/env python
"""
FlexConsensus CLI test scenarios.

FlexConsensus receives N latent spaces with the SAME number of points and an
ordered point-to-point correspondence (point i in every space is the same
particle). It learns a common consensus space and, at predict time, reports for
each space a per-point consensus error and representation error, revealing where
the spaces agree or disagree.

The phantom is a set of N synthetic spaces built from a shared, ordered 1-D
coordinate (a circle): each space is a different random linear embedding of that
coordinate (different dimensionality), so the spaces *agree* almost everywhere —
except a contiguous region where one space follows a *different* trend (a rotated
coordinate), i.e. a built-in disagreement region.

Coverage:

* ``--input_space``  : N spaces via the ``NAME:path`` convention
* ``--lat_dim``      : default (min of input dims) and explicit
* ``--mode``         : train, predict (+ ``--reload``), send_to_pickle
* ``--epochs`` / ``--batch_size`` / ``--learning_rate``
* outputs: a trained ``FlexConsensus`` model and the per-space
  ``*_consensus.npy`` / ``*_consensus_error.npy`` / ``*_representation_error.npy``

Both input conventions are exercised: the ``NAME:path`` form and plain paths.
Plain paths used to crash (``input_spaces_name`` stayed ``None`` and was zipped in
the train-time TensorBoard embedding and the predict save loop); this is now fixed
by falling back to the network's own space names, and the ``*_plain`` scenarios
guard that fix.
"""

import os
import numpy as np

from common import Scenario

M = 512                       # points per space (shared, ordered correspondence)
DIMS = (4, 5, 6)              # per-space dimensionality (different on purpose)
NAMES = ("A", "B", "C")


def prepare_data(workdir):
    """Write N ordered, corresponding latent spaces (agree everywhere except a
    built-in disagreement region in the first space)."""
    data_dir = os.path.join(workdir, "data")
    os.makedirs(data_dir, exist_ok=True)
    rng = np.random.default_rng(0)

    # Shared, ordered 1-D coordinate embedded on a circle -> (M, 2)
    t = np.linspace(0.0, 1.0, M)
    u = np.stack([np.cos(2 * np.pi * t), np.sin(2 * np.pi * t)], axis=1).astype(np.float32)

    # Disagreement region (contiguous block) for the first space: rotated trend
    a, b = M // 3, M // 3 + M // 6
    th = np.pi / 2.0
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], dtype=np.float32)

    paths = {}
    for k, (D, name) in enumerate(zip(DIMS, NAMES)):
        W = rng.standard_normal((2, D)).astype(np.float32)
        u_k = u.copy()
        if k == 0:
            u_k[a:b] = u[a:b] @ R.T            # first space disagrees in [a, b]
        s = (u_k @ W + 0.03 * rng.standard_normal((M, D))).astype(np.float32)
        p = os.path.join(data_dir, f"space_{name}.npy")
        np.save(p, s)
        paths[name] = p
    return {"paths": paths, "disagree": (a, b)}


def data_checks(workdir):
    """Confirm the N spaces share the point correspondence (same #points)."""
    data_dir = os.path.join(workdir, "data")
    shapes = [np.load(os.path.join(data_dir, f"space_{n}.npy")).shape for n in NAMES]
    same_n = len({s[0] for s in shapes}) == 1
    return [(
        "spaces_share_correspondence",
        bool(same_n and len(shapes) == len(NAMES)),
        f"{len(shapes)} spaces, shapes {shapes} (all share {shapes[0][0]} ordered points)")]


def _named(paths):
    """Build the NAME:path arguments for --input_space."""
    return [f"{name}:{paths[name]}" for name in NAMES]


def _plain(paths):
    """Build plain-path arguments for --input_space (auto-named Input_00.. by the program)."""
    return [paths[name] for name in NAMES]


def scenarios(workdir, data):
    runs = os.path.join(workdir, "runs")
    paths = data["paths"]
    named = _named(paths)

    def out(name):
        return os.path.join(runs, name)

    def mkdirp(path):
        return lambda: os.makedirs(path, exist_ok=True)

    scn = []

    # 1) train | default lat_dim (= min input dim)
    scn.append(Scenario(
        name="train_default",
        description="train | 3 spaces (NAME:path) | default lat_dim",
        program="flexconsensus",
        args=["--input_space", *named, "--mode", "train",
              "--epochs", "5", "--batch_size", "128",
              "--output_path", out("train_default")],
        expect_files=[os.path.join(out("train_default"), "FlexConsensus")],
        timeout=600))

    # 2) predict | reload model from (1) -> consensus + error files per space
    scn.append(Scenario(
        name="predict",
        description="predict | --reload train_default -> *_consensus / *_error per space",
        program="flexconsensus",
        args=["--input_space", *named, "--mode", "predict",
              "--reload", out("train_default"), "--batch_size", "128",
              "--output_path", out("predict")],
        expect_files=[os.path.join(out("predict"), f"{n}_consensus.npy") for n in NAMES]
                     + [os.path.join(out("predict"), f"{n}_consensus_error.npy") for n in NAMES]
                     + [os.path.join(out("predict"), f"{n}_representation_error.npy") for n in NAMES],
        pre=mkdirp(out("predict")),
        timeout=600))

    # 3) train | explicit lat_dim + custom epochs/batch/lr
    scn.append(Scenario(
        name="train_custom",
        description="train | explicit --lat_dim 3 | epochs=4 batch=256 lr=1e-4",
        program="flexconsensus",
        args=["--input_space", *named, "--mode", "train", "--lat_dim", "3",
              "--epochs", "4", "--batch_size", "256", "--learning_rate", "1e-4",
              "--output_path", out("train_custom")],
        expect_files=[os.path.join(out("train_custom"), "FlexConsensus")],
        timeout=600))

    # 4) train | PLAIN paths (auto-named) — guards the input_spaces_name=None fix
    scn.append(Scenario(
        name="train_plain",
        description="train | 3 spaces (plain paths, auto-named) | default lat_dim",
        program="flexconsensus",
        args=["--input_space", *_plain(paths), "--mode", "train",
              "--epochs", "5", "--batch_size", "128",
              "--output_path", out("train_plain")],
        expect_files=[os.path.join(out("train_plain"), "FlexConsensus")],
        timeout=600))

    # 5) predict | PLAIN paths, reload (4) — auto-named outputs Input_00.. per space
    scn.append(Scenario(
        name="predict_plain",
        description="predict | plain paths | --reload train_plain (auto-named outputs)",
        program="flexconsensus",
        args=["--input_space", *_plain(paths), "--mode", "predict",
              "--reload", out("train_plain"), "--batch_size", "128",
              "--output_path", out("predict_plain")],
        expect_files=[os.path.join(out("predict_plain"), f"Input_{i:02d}_consensus.npy")
                      for i in range(len(NAMES))],
        pre=mkdirp(out("predict_plain")),
        timeout=600))

    # 6) send_to_pickle (currently a no-op for flexconsensus — must still exit cleanly)
    scn.append(Scenario(
        name="send_to_pickle",
        description="send_to_pickle (no-op smoke check)",
        program="flexconsensus",
        args=["--input_space", *named, "--mode", "send_to_pickle",
              "--output_path", out("send_to_pickle")],
        expect_files=[],
        pre=mkdirp(out("send_to_pickle")),
        timeout=300))

    return scn
