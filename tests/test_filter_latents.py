#!/usr/bin/env python
"""
filter_latents CLI test scenarios.

``filter_latents`` is a pure latent-space post-processing step: it loads a latent
space from a ``.npy`` file, computes a per-vector z-score from the mean distance
to its ``--n_neighbours`` nearest neighbours, and keeps (or returns the ids of)
the vectors whose z-score is below ``--thr``. It needs neither a network nor
images.

The phantom here is a synthetic latent cloud: a Gaussian bulk plus a handful of
clear outliers, so the filtering is deterministic and observable (the outliers
are the ones removed).

Coverage:

* ``--latents`` (input .npy)
* ``--thr`` (z-score threshold) and ``--n_neighbours`` (varied)
* ``--return_ids`` (return indices instead of the filtered space)
* ``--batch_size``
* output: ``filtered_latents.npy``

Note: ``filter_latents`` does not create its ``--output_path`` (it ``np.save``s
straight into it); the suite pre-creates it.
"""

import os
import numpy as np

from common import Scenario

N_MAIN = 270         # Gaussian bulk
N_OUTLIERS = 30      # clear outliers (removed by the filter)
DIM = 8


def prepare_data(workdir):
    """Write a synthetic latent space (bulk + outliers) to a .npy file."""
    data_dir = os.path.join(workdir, "data")
    os.makedirs(data_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    bulk = rng.standard_normal((N_MAIN, DIM)).astype(np.float32)
    outliers = (rng.standard_normal((N_OUTLIERS, DIM))
                + rng.choice([-1.0, 1.0], (N_OUTLIERS, DIM)) * 6.0).astype(np.float32)
    latents = np.concatenate([bulk, outliers], axis=0)
    path = os.path.join(data_dir, "latents.npy")
    np.save(path, latents)
    return {"latents": path}


def data_checks(workdir):
    """Confirm the synthetic latent space carries detectable outliers."""
    path = os.path.join(workdir, "data", "latents.npy")
    lat = np.load(path)
    # Outlier rows have a much larger norm than the Gaussian bulk.
    norms = np.linalg.norm(lat, axis=1)
    bulk_max = float(np.median(norms[:N_MAIN]))
    out_min = float(norms[N_MAIN:].min())
    return [(
        "latents_have_outliers",
        bool(out_min > 2.0 * bulk_max),
        f"outlier min-norm {out_min:.1f} >> bulk median-norm {bulk_max:.1f}")]


def scenarios(workdir, data):
    runs = os.path.join(workdir, "runs")
    latents = data["latents"]

    def out(name):
        return os.path.join(runs, name)

    def mkdirp(path):
        return lambda: os.makedirs(path, exist_ok=True)

    def filtered(name):
        return [os.path.join(out(name), "filtered_latents.npy")]

    scn = []

    # 1) default filtering (thr=1.0, n_neighbours=10) -> filtered space
    scn.append(Scenario(
        name="filter_default",
        description="filter | default thr=1.0 n_neighbours=10 -> filtered space",
        program="filter_latents",
        args=["--latents", latents, "--output_path", out("filter_default")],
        expect_files=filtered("filter_default"),
        pre=mkdirp(out("filter_default")),
        timeout=300))

    # 2) return ids instead of the filtered space
    scn.append(Scenario(
        name="filter_return_ids",
        description="filter | --return_ids -> kept indices",
        program="filter_latents",
        args=["--latents", latents, "--return_ids",
              "--output_path", out("filter_return_ids")],
        expect_files=filtered("filter_return_ids"),
        pre=mkdirp(out("filter_return_ids")),
        timeout=300))

    # 3) custom threshold / neighbours / batch size
    scn.append(Scenario(
        name="filter_custom",
        description="filter | thr=2.0 n_neighbours=5 batch_size=32",
        program="filter_latents",
        args=["--latents", latents, "--thr", "2.0", "--n_neighbours", "5",
              "--batch_size", "32", "--output_path", out("filter_custom")],
        expect_files=filtered("filter_custom"),
        pre=mkdirp(out("filter_custom")),
        timeout=300))

    return scn
