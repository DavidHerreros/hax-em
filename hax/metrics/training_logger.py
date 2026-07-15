#!/usr/bin/env python
"""Logging cadence and background offload for the training programs.

The things a training program logs differ in cost by orders of magnitude (on a
box-128 HetSIREN run: a loss scalar is free, a set of volume slices costs ~0.07 s,
a latent-space embedding costs ~8 s), so a single ``--log_every`` would either
throttle the cheap things needlessly or leave the expensive ones running too
often. ``TrainingLogger`` gives each *tier* its own cadence:

* ``scalars``    — training losses. A few times per epoch.
* ``images``     — predicted images and volume slices. Every ``image_every`` epochs.
* ``landscape``  — latent embedding, intermediate volumes, angular plots. Every
                   ``landscape_every`` epochs; this is the expensive tier.
* ``checkpoint`` — intermediate checkpoint. Every ``checkpoint_every`` epochs.
                   Kept separate from ``landscape`` so how often you can resume is
                   not tied to how often you want pictures.

On top of the cadence, the logger moves the *host-side* part of the work (TensorBoard
writes, ``.mrc`` writes, matplotlib figures) onto a single background thread, so it
overlaps with the next epoch's GPU work instead of stalling it. Submit only work whose
inputs will not be mutated afterwards — JAX arrays are immutable, so snapshots of
``(graphdef, state)`` and decoded volumes are safe to hand over.

``time_budget`` (a fraction of wall clock, 0 = off) is an optional self-tuning guard:
a tier is skipped when logging has already eaten more than its share of the run. It
adapts to dataset and box size on its own, which epoch counts cannot do.
"""

import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

from hax.utils import bcolors


class TrainingLogger:

    TIERS = ("scalars", "images", "landscape", "checkpoint")

    def __init__(self, image_every=1, landscape_every=5, checkpoint_every=5,
                 scalars_per_epoch=10, steps_per_epoch=1, time_budget=0.0, background=True):
        self.image_every = int(image_every)
        self.landscape_every = int(landscape_every)
        self.checkpoint_every = int(checkpoint_every)
        self.steps_per_epoch = max(1, int(steps_per_epoch))
        self.scalars_per_epoch = max(1, int(scalars_per_epoch))
        self.scalar_every = max(1, self.steps_per_epoch // self.scalars_per_epoch)
        self.time_budget = float(time_budget)

        self._executor = ThreadPoolExecutor(max_workers=1) if background else None
        self._futures = []
        self._t_start = time.perf_counter()
        self._spent = 0.0            # wall clock spent logging *on the training thread*
        self._skipped = {tier: 0 for tier in self.TIERS}

    # ------------------------------------------------------------------ cadence
    def _every(self, tier):
        return {"images": self.image_every,
                "landscape": self.landscape_every,
                "checkpoint": self.checkpoint_every}[tier]

    def should(self, tier, epoch):
        """True when ``tier`` is due at the start of ``epoch`` (0-based)."""
        every = self._every(tier)
        if every <= 0:                      # tier disabled
            return False
        if epoch <= 0:                      # nothing meaningful to show yet
            return tier == "images"         # ...except the images, which are the cheap sanity check
        if epoch % every != 0:
            return False
        if not self._within_budget():
            self._skipped[tier] += 1
            return False
        return True

    def should_log_scalars(self, step):
        return step % self.scalar_every == 0

    def _within_budget(self):
        if self.time_budget <= 0.0:
            return True
        elapsed = time.perf_counter() - self._t_start
        if elapsed <= 0.0:
            return True
        return (self._spent / elapsed) < self.time_budget

    # ------------------------------------------------------------------ timing
    def start(self):
        """Mark the beginning of the run (call right before the training loop)."""
        self._t_start = time.perf_counter()
        return self

    class _Section:
        def __init__(self, logger):
            self._logger = logger

        def __enter__(self):
            self._t0 = time.perf_counter()
            return self

        def __exit__(self, *exc):
            self._logger._spent += time.perf_counter() - self._t0
            return False

    def section(self, tier=None):
        """Charge the enclosed block to the logging time budget."""
        return TrainingLogger._Section(self)

    # ------------------------------------------------------------------ offload
    def submit(self, fn, *args, **kwargs):
        """Run host-side logging work off the training thread (FIFO, one worker)."""
        if self._executor is None:
            with self.section():
                return fn(*args, **kwargs)

        future = self._executor.submit(fn, *args, **kwargs)
        future.add_done_callback(self._report_failure)
        self._futures = [f for f in self._futures if not f.done()] + [future]
        return future

    @staticmethod
    def _report_failure(future):
        exc = future.exception()
        if exc is not None:
            print(f"{bcolors.WARNING}\nBackground logging task failed "
                  f"({type(exc).__name__}: {exc}); training continues.{bcolors.ENDC}",
                  file=sys.stderr)
            traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)

    def close(self):
        """Wait for pending background writes (call once training is done)."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        skipped = {t: n for t, n in self._skipped.items() if n}
        if skipped:
            detail = ", ".join(f"{t} x{n}" for t, n in skipped.items())
            print(f"{bcolors.WARNING}Logging time budget ({self.time_budget:.0%} of wall clock) "
                  f"skipped: {detail}.{bcolors.ENDC}")

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.close()
        return False
