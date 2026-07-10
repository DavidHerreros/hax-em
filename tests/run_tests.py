#!/usr/bin/env python
"""
Hax CLI test-suite runner.

Fast, data-free smoke tests for the programs exposed by ``hax_project_manager``.
A synthetic phantom with continuous + compositional conformational heterogeneity
is generated on the fly (different conformation, pose, shift and CTF per
particle), then every program is driven through its **real CLI entry point**
across a broad sweep of options, so future breakages show up as failing
scenarios.

Usage
-----
    # equivalent in spirit to:  hax_project_manager --gpu 0 hetsiren test
    python tests/run_tests.py --gpu 0 hetsiren

    python tests/run_tests.py --gpu 0 all          # every implemented program
    python tests/run_tests.py --gpu 0 --quick hetsiren   # skip slow (fit_volume) cases
    python tests/run_tests.py --gpu 0 --keep hetsiren    # keep the work dir for inspection

Notes
-----
* Run it with the Python of an environment where ``hax`` imports and runs
  (flax 0.12, jax 0.9, xmipp_metadata, cuml).  The same interpreter is used to
  spawn the CLI subprocesses unless ``--python`` is given.
* This runner generates the phantom with JAX on the CPU (so it does not hold GPU
  memory while the CLI subprocesses train); the subprocesses themselves use the
  GPU selected with ``--gpu``.
* It does NOT modify or import any Hax *program* module — every program is run
  exactly as a user would via the CLI.
"""

import os
import sys
import argparse
import importlib
import shutil
import tempfile

# Generate the phantom on CPU so the runner process holds no GPU memory.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import run_suite, print_header, GREEN, RED, YELLOW, BOLD, END  # noqa: E402

# Registry of programs that have a test module ``test_<name>.py`` exposing
# prepare_data(workdir) / data_checks(workdir) / scenarios(workdir, data) / SLOW.
PROGRAMS = ["preprocess_particles", "preprocess_volumes",
            "hetsiren", "reconsiren", "reconsiren_het_only", "zernike3deep",
            "estimate_latent_covariances", "latent_space_deconvolution",
            "filter_latents", "flexconsensus", "decode_states_from_latents",
            "modart", "annotate_space", "image_gray_scale_adjustment",
            "volume_gray_scale_adjustment", "display_metrics"]

# Programs whose *training* has a large, box/batch-independent GPU footprint
# (multi-hypothesis pose rendering + a big heterogeneity decoder + several
# optimizers). They OOM on GPUs with only a few GB free — e.g. the ReconSIREN
# train step overflows an 8 GB card even at box=16 / batch_size=1, whereas
# HetSIREN trains fine on the same card. Pass ``--skip-heavy`` to drop them from
# an ``all`` run when testing on a small GPU.
HEAVY = ["reconsiren", "reconsiren_het_only"]


def run_program(program, gpu, workdir, quick, python):
    mod = importlib.import_module(f"test_{program}")
    print_header(f"PROGRAM: {program}")

    # --- Phantom generation -------------------------------------------------
    print(f"  generating phantom dataset(s) in {workdir} …", flush=True)
    data = mod.prepare_data(workdir)

    # --- Data integrity checks ---------------------------------------------
    rc_checks = 0
    if hasattr(mod, "data_checks"):
        print(f"\n  {BOLD}Data integrity checks{END}")
        for name, ok, detail in mod.data_checks(workdir):
            tag = f"{GREEN}OK  {END}" if ok else f"{RED}FAIL{END}"
            print(f"    {tag} {name} — {detail}")
            rc_checks |= (0 if ok else 1)

    # --- CLI scenarios ------------------------------------------------------
    scn = mod.scenarios(workdir, data)
    slow = getattr(mod, "SLOW", set())
    if quick:
        for s in scn:
            if s.name in slow:
                s.skip = True
                s.skip_reason = "slow (fit_volume) — skipped by --quick"

    print(f"\n  {BOLD}Running {len(scn)} CLI scenario(s)"
          f"{' (quick mode)' if quick else ''}{END}\n")
    rc = run_suite(scn, gpu, python)
    return rc | rc_checks


def main():
    parser = argparse.ArgumentParser(
        description="Run the Hax CLI smoke-test suite on a synthetic phantom.")
    parser.add_argument("program", help="Program to test, or 'all'. "
                        f"Implemented: {', '.join(PROGRAMS)}")
    parser.add_argument("--gpu", default=None,
                        help="GPU id passed through to each CLI run (CUDA_VISIBLE_DEVICES).")
    parser.add_argument("--workdir", default=None,
                        help="Where to put phantom data + run outputs "
                             "(default: a temp dir, removed unless --keep).")
    parser.add_argument("--quick", action="store_true",
                        help="Skip slow scenarios (those needing fit_volume).")
    parser.add_argument("--skip-heavy", action="store_true",
                        help="Skip GPU-memory-heavy programs (%s) — their training "
                             "OOMs on small GPUs. Only affects the 'all' target; a "
                             "heavy program named explicitly still runs."
                             % ", ".join(HEAVY))
    parser.add_argument("--keep", action="store_true",
                        help="Keep the work directory after running.")
    parser.add_argument("--python", default=sys.executable,
                        help="Python interpreter used to spawn the CLI subprocesses "
                             "(default: the one running this script).")
    args = parser.parse_args()

    if args.program == "all":
        programs = list(PROGRAMS)
        if args.skip_heavy:
            skipped = [p for p in programs if p in HEAVY]
            programs = [p for p in programs if p not in HEAVY]
            if skipped:
                print(f"  {YELLOW}--skip-heavy: skipping {', '.join(skipped)} "
                      f"(GPU-memory-heavy){END}")
    elif args.program in PROGRAMS:
        # An explicitly named program always runs, even if heavy — --skip-heavy
        # only prunes the 'all' target.
        programs = [args.program]
    else:
        parser.error(f"Unknown program '{args.program}'. "
                     f"Implemented: {', '.join(PROGRAMS)}, or 'all'.")

    workdir = args.workdir or tempfile.mkdtemp(prefix="hax_test_")
    os.makedirs(workdir, exist_ok=True)

    rc = 0
    try:
        for program in programs:
            rc |= run_program(program, args.gpu, os.path.join(workdir, program),
                              args.quick, args.python)
    finally:
        if args.keep:
            print(f"  {YELLOW}work dir kept at: {workdir}{END}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)

    if rc == 0:
        print(f"{GREEN}{BOLD}ALL TESTS PASSED{END}\n")
    else:
        print(f"{RED}{BOLD}SOME TESTS FAILED{END}\n")
    sys.exit(rc)


if __name__ == "__main__":
    main()
