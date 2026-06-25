#!/usr/bin/env python
"""
Shared utilities for the Hax CLI test-suite.

The suite is a *black-box smoke test*: every scenario is executed through the
real ``hax.cli:main`` entry point in a **fresh subprocess** — exactly the code
path of the installed ``hax_project_manager`` console script — so that argument
parsing, ``--gpu`` handling and the full program run are all exercised the way a
user would run them.  Nothing here imports or patches any Hax *program* module.

A scenario passes when the subprocess exits 0 *and* every expected output file
exists.  Stdout/stderr are captured; the tail is shown on failure.  A heuristic
NaN-in-loss warning is surfaced (informational only) since it can hint at a
numerical regression without being a hard error.
"""

import os
import sys
import time
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional, Callable

# ANSI colours (kept local so the suite has no Hax import dependency at module load)
GREEN, RED, YELLOW, CYAN, BOLD, END = (
    "\033[92m", "\033[91m", "\033[93m", "\033[96m", "\033[1m", "\033[0m"
)

# The one-liner that reproduces the `hax_project_manager` console entry point.
_CLI_BOOTSTRAP = "from hax.cli import main; main()"


@dataclass
class Scenario:
    """A single test invocation.

    By default this runs ``program`` through the real ``hax.cli:main`` entry point
    (like ``hax_project_manager``). If ``script`` is set, that Python script is run
    directly instead (used for headless component tests of GUI programs that can't
    be driven through the blocking CLI).
    """
    name: str
    description: str
    program: str                       # e.g. "hetsiren" (label only when script is set)
    args: List[str]                    # CLI args after the program name (or for the script)
    expect_files: List[str] = field(default_factory=list)  # absolute paths
    timeout: int = 1800
    pre: Optional[Callable[[], None]] = None   # optional setup hook (e.g. mkdir)
    skip: bool = False
    skip_reason: str = ""
    script: Optional[str] = None       # run this script directly instead of the CLI


@dataclass
class Result:
    scenario: Scenario
    ok: bool
    exit_code: int
    duration: float
    missing: List[str]
    nan_loss: bool
    tail: str


def run_scenario(sc: Scenario, gpu: Optional[str], python: Optional[str] = None) -> Result:
    """Run one scenario as a subprocess and classify the outcome."""
    python = python or sys.executable
    if sc.pre is not None:
        sc.pre()

    # The parent may run JAX on CPU (for phantom generation); the subprocess must
    # NOT inherit that — the CLI picks its GPU from --gpu, while a direct script
    # gets CUDA_VISIBLE_DEVICES set explicitly below.
    env = {k: v for k, v in os.environ.items() if k != "JAX_PLATFORMS"}

    if sc.script is not None:
        cmd = [python, sc.script] + sc.args
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    else:
        cmd = [python, "-c", _CLI_BOOTSTRAP]
        if gpu is not None:
            cmd += ["--gpu", str(gpu)]
        cmd += [sc.program] + sc.args

    t0 = time.time()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, timeout=sc.timeout, env=env)
    dt = time.time() - t0

    out = proc.stdout or ""
    missing = [f for f in sc.expect_files if not os.path.exists(f)]
    nan_loss = "loss=nan" in out or "loss=inf" in out
    # A NaN/Inf loss is treated as a failure: the known source (empty-mask masked
    # mean in the negative-density L1 term) is fixed, so its reappearance is a
    # regression worth catching.
    ok = (proc.returncode == 0) and not missing and not nan_loss

    # Keep a compact tail for diagnostics (filter noisy progress-bar lines)
    lines = [ln for ln in out.splitlines()
             if "it/s" not in ln and not ln.strip().endswith("it]") and ln.strip()]
    tail = "\n".join(lines[-25:])

    return Result(sc, ok, proc.returncode, dt, missing, nan_loss, tail)


def print_header(title: str):
    print(f"\n{CYAN}{BOLD}{'=' * 78}{END}")
    print(f"{CYAN}{BOLD}  {title}{END}")
    print(f"{CYAN}{BOLD}{'=' * 78}{END}\n")


def print_result(idx: int, total: int, res: Result):
    sc = res.scenario
    if sc.skip:
        print(f"  [{idx}/{total}] {YELLOW}SKIP{END} {sc.name} — {sc.skip_reason}")
        return
    status = f"{GREEN}PASS{END}" if res.ok else f"{RED}FAIL{END}"
    print(f"  [{idx}/{total}] {status} {BOLD}{sc.name}{END}  ({res.duration:.0f}s)")
    print(f"          {sc.description}")
    if not res.ok:
        print(f"          {RED}exit={res.exit_code}{END}", end="")
        if res.nan_loss:
            print(f"  {RED}NaN/Inf loss{END}", end="")
        if res.missing:
            print(f"  {RED}missing outputs: {', '.join(res.missing)}{END}", end="")
        print()
        if res.tail:
            indented = "\n".join("            " + ln for ln in res.tail.splitlines())
            print(f"{RED}{indented}{END}")


def run_suite(scenarios: List[Scenario], gpu: Optional[str], python: Optional[str] = None) -> int:
    """Run a list of scenarios, print a report, return a process exit code."""
    total = len(scenarios)
    results: List[Result] = []
    for i, sc in enumerate(scenarios, 1):
        if sc.skip:
            print_result(i, total, Result(sc, True, 0, 0.0, [], False, ""))
            continue
        print(f"  [{i}/{total}] {BOLD}{sc.name}{END} … running", flush=True)
        try:
            res = run_scenario(sc, gpu, python)
        except subprocess.TimeoutExpired:
            res = Result(sc, False, -1, sc.timeout, sc.expect_files, False,
                         f"TIMEOUT after {sc.timeout}s")
        results.append(res)
        # reprint the line with the verdict
        print("\033[F\033[K", end="")   # move up + clear the "running" line
        print_result(i, total, res)

    n_pass = sum(1 for r in results if r.ok)
    n_fail = sum(1 for r in results if not r.ok)
    n_skip = sum(1 for s in scenarios if s.skip)

    print_header("SUMMARY")
    print(f"  {GREEN}passed : {n_pass}{END}")
    print(f"  {RED}failed : {n_fail}{END}")
    print(f"  {YELLOW}skipped: {n_skip}{END}")
    if n_fail:
        print(f"\n  {RED}{BOLD}FAILURES:{END}")
        for r in results:
            if not r.ok:
                print(f"    - {r.scenario.name} (exit {r.exit_code})")
    print()
    return 1 if n_fail else 0
