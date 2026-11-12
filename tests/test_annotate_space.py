#!/usr/bin/env python
"""
annotate_space (headless) test scenarios.

annotate_space is an interactive PyQt5 + napari desktop app that talks to ChimeraX
over a socket — it cannot be driven through the CLI like the batch programs (its
``main()`` blocks on ``QApplication.exec_()`` and needs a display). Instead, this
suite exercises its *headless, logic-bearing* components directly, via the helper
script ``annotate_space_headless.py`` (run as a subprocess per check):

  * dimred     — PCA + UMAP dimensionality reduction (``DimRedQThread``)
  * clustering — KMeans + along-dimension clustering (``ClusteringQThread``)
  * socket     — viewer_socket ``Server``<->``Client`` round-trip (FromFiles map copy)
  * utils      — pure helpers (getImagePath / getServerProgram / ...)
  * offscreen  — [Tier 2, best-effort] assemble the napari viewer + the app's
                 ``MultipleViewerWidget`` under ``QT_QPA_PLATFORM=offscreen``, then
                 close. Skips gracefully (still exit 0) if offscreen isn't usable.

Out of scope (genuinely needs a GUI / ChimeraX / manual interaction): the napari
canvas, lasso/point selection, ChimeraX morphing, and screenshots.
"""

import os

from common import Scenario

_HEADLESS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "annotate_space_headless.py")

CHECKS = [
    ("dimred", "PCA + UMAP dimensionality reduction (DimRedQThread)"),
    ("clustering", "KMeans + along-dimension clustering (ClusteringQThread)"),
    ("socket", "viewer_socket Server<->Client round-trip (FromFiles map copy)"),
    ("utils", "pure helpers (getImagePath / getServerProgram / ...)"),
    ("offscreen", "[Tier 2] napari viewer + MultipleViewerWidget assembled offscreen"),
]


def prepare_data(workdir):
    os.makedirs(workdir, exist_ok=True)
    return {}


def scenarios(workdir, data):
    runs = os.path.join(workdir, "runs")
    scn = []
    for check, desc in CHECKS:
        wd = os.path.join(runs, check)
        scn.append(Scenario(
            name=f"annotate_space:{check}",
            description=desc,
            program="annotate_space",
            script=_HEADLESS,
            args=["--check", check, "--workdir", wd],
            pre=(lambda p=wd: os.makedirs(p, exist_ok=True)),
            timeout=600))
    return scn
