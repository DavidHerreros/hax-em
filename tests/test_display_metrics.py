#!/usr/bin/env python
"""
display_metrics (headless) test scenarios.

``display_metrics`` (hax/metrics/writer.py) is a convenience launcher: it starts
TensorBoard on a ``--logdir`` and blocks until Ctrl+C. It cannot be driven to
completion through the CLI like the batch programs (it never returns), so — as
with ``annotate_space`` — this suite exercises its logic-bearing parts through
the helper script ``display_metrics_headless.py`` (one subprocess per check):

  * writer  — ``JaxSummaryWriter`` logs JAX-array scalars (auto-converted to
              numpy by its ``__getattribute__`` wrapper) and central volume
              slices; asserts a ``tfevents`` file is written.
  * launch  — replicate ``main()``'s ``tensorboard.program`` launch in-process
              and HTTP-probe the returned URL (server actually serves the run).
  * cli     — run the genuine ``display_metrics --logdir`` entry point through
              ``hax.cli:main``, wait for the "TensorBoard is running at <url>"
              line, HTTP-probe it, then SIGINT and assert a clean shutdown.

Out of scope (genuinely needs a browser): visually inspecting the TensorBoard UI.
"""

import os

from common import Scenario

_HEADLESS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "display_metrics_headless.py")

CHECKS = [
    ("writer", "JaxSummaryWriter logs JAX arrays + volume slices -> tfevents"),
    ("launch", "tensorboard.program launch + HTTP probe of served run"),
    ("cli", "display_metrics --logdir through hax.cli:main, SIGINT shutdown"),
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
            name=f"display_metrics:{check}",
            description=desc,
            program="display_metrics",
            script=_HEADLESS,
            args=["--check", check, "--workdir", wd],
            pre=(lambda p=wd: os.makedirs(p, exist_ok=True)),
            timeout=300))
    return scn
