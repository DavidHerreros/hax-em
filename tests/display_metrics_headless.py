#!/usr/bin/env python
"""
Headless component tests for the ``display_metrics`` program (hax/metrics/writer.py).

``display_metrics`` is a thin convenience wrapper that launches TensorBoard on a
log directory and then blocks until Ctrl+C — there is nothing to *look* at
programmatically, but every moving part can be exercised without a human opening
a browser:

  * writer  — the real library logic in ``JaxSummaryWriter``: the
              ``__getattribute__`` wrapper that auto-converts JAX arrays to numpy
              before handing them to ``tensorboardX``, plus the jitted
              ``add_volumes_slices`` (MAD/mean, low-pass, color map, central
              slices). Asserts a ``tfevents`` file is actually written.
  * launch  — replicate what ``main()`` does (``tensorboard.program`` →
              ``tb.launch()`` → URL) in-process and HTTP-probe the URL to confirm
              the server actually serves the run. No blocking loop.
  * cli     — drive the genuine entry point: spawn ``display_metrics --logdir``
              through ``hax.cli:main`` (exactly as ``hax_project_manager`` would),
              wait for the "TensorBoard is running at <url>" line, HTTP-probe it,
              then send SIGINT and assert the clean "Received Ctrl+C" shutdown.

Each check prints ``PASS``/``FAIL``/``SKIP`` and the script exits non-zero if any
check fails, so it slots into the suite as a ``Scenario(script=...)``.
"""

import argparse
import glob
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.request

# TensorBoard / werkzeug are noisy on stderr; quiet them down.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

GREEN, RED, YELLOW, END = "\033[92m", "\033[91m", "\033[93m", "\033[0m"

_ANSI = re.compile(r"\033\[[0-9;]*m")
_URL = re.compile(r"https?://[^\s\033]+")


def _ok(name, detail=""):
    print(f"  {GREEN}PASS{END} {name}" + (f" — {detail}" if detail else ""))
    return True


def _fail(name, detail=""):
    print(f"  {RED}FAIL{END} {name}" + (f" — {detail}" if detail else ""))
    return False


def _skip(name, detail=""):
    print(f"  {YELLOW}SKIP{END} {name}" + (f" — {detail}" if detail else ""))
    return True


def _http_ok(url, timeout=15.0):
    """Return True if a GET on *url* returns HTTP 200 within *timeout*."""
    deadline = time.time() + timeout
    last = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return True, f"HTTP {resp.status}"
                last = f"HTTP {resp.status}"
        except Exception as exc:  # connection refused while the server warms up
            last = repr(exc)
            time.sleep(0.5)
    return False, last


# --------------------------------------------------------------------------- #
# writer — JaxSummaryWriter library logic                                      #
# --------------------------------------------------------------------------- #
def check_writer(workdir):
    name = "writer (JaxSummaryWriter logs JAX arrays + volume slices)"
    logdir = os.path.join(workdir, "logs")
    os.makedirs(logdir, exist_ok=True)

    import jax.numpy as jnp
    from hax.metrics.writer import JaxSummaryWriter

    writer = JaxSummaryWriter(log_dir=logdir)

    # Scalars passed as *JAX* arrays — exercises the __getattribute__ wrapper that
    # converts JaxArray -> numpy before tensorboardX sees them.
    for step in range(5):
        writer.add_scalar("loss", jnp.asarray(1.0 / (step + 1)), step)
        writer.add_scalar("plain_python_float", float(step), step)  # no-conversion path

    # Central-slice volume logging (jitted color map + add_image/add_text).
    volumes = jnp.stack([
        jnp.zeros((16, 16, 16)),
        jnp.linspace(0.0, 1.0, 16 ** 3).reshape(16, 16, 16),
        jnp.ones((16, 16, 16)),
    ], axis=0)
    writer.add_volumes_slices(volumes)
    writer.flush()
    writer.close()

    events = glob.glob(os.path.join(logdir, "**", "*tfevents*"), recursive=True)
    if not events:
        return _fail(name, "no tfevents file written")
    sizes = [os.path.getsize(e) for e in events]
    if not any(s > 0 for s in sizes):
        return _fail(name, "tfevents file is empty")
    return _ok(name, f"{len(events)} event file(s), {max(sizes)} bytes")


def _make_logdir(workdir):
    """Write a tiny run so TensorBoard has something to serve."""
    logdir = os.path.join(workdir, "logs")
    os.makedirs(logdir, exist_ok=True)
    if not glob.glob(os.path.join(logdir, "**", "*tfevents*"), recursive=True):
        import jax.numpy as jnp
        from hax.metrics.writer import JaxSummaryWriter
        w = JaxSummaryWriter(log_dir=logdir)
        for step in range(5):
            w.add_scalar("loss", jnp.asarray(1.0 / (step + 1)), step)
        w.flush()
        w.close()
    return logdir


# --------------------------------------------------------------------------- #
# launch — in-process TensorBoard launch (what main() does, minus the block)   #
# --------------------------------------------------------------------------- #
def check_launch(workdir):
    name = "launch (tensorboard.program.launch + HTTP probe)"
    logdir = _make_logdir(workdir)
    try:
        from tensorboard import program
    except Exception as exc:
        return _skip(name, f"tensorboard not importable: {exc!r}")

    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", logdir, "--port", "0"])
    try:
        url = tb.launch()
    except Exception as exc:
        return _fail(name, f"tb.launch() raised {exc!r}")

    if not _URL.match(url or ""):
        return _fail(name, f"launch returned no usable URL: {url!r}")

    ok, detail = _http_ok(url)
    return (_ok if ok else _fail)(name, f"{url} ({detail})")


# --------------------------------------------------------------------------- #
# cli — the genuine entry point through hax.cli:main, killed with SIGINT       #
# --------------------------------------------------------------------------- #
def check_cli(workdir):
    name = "cli (display_metrics --logdir through hax.cli:main, SIGINT shutdown)"
    logdir = _make_logdir(workdir)

    # Same code path as `hax_project_manager display_metrics --logdir <dir>`.
    # ``-u`` keeps the child's stdout unbuffered so the "running at <url>" line
    # reaches the reader thread immediately (otherwise it sits in the child's
    # block-buffered pipe until exit and the URL is only seen after shutdown).
    cmd = [sys.executable, "-u", "-c", "from hax.cli import main; main()",
           "display_metrics", "--logdir", logdir]
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, env=env)

    captured = []

    def _reader():
        for line in proc.stdout:
            captured.append(line)

    t = threading.Thread(target=_reader, daemon=True)
    t.start()

    # Wait for the "TensorBoard is running at <url>" announcement.
    url = None
    deadline = time.time() + 90
    while time.time() < deadline and proc.poll() is None:
        for line in list(captured):
            if "TensorBoard is running at" in line:
                m = _URL.search(_ANSI.sub("", line))
                if m:
                    url = m.group(0)
                    break
        if url:
            break
        time.sleep(0.3)

    if url is None:
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
        return _fail(name, "never printed a TensorBoard URL\n" +
                     "".join(captured)[-800:])

    http_ok, http_detail = _http_ok(url)

    # Ctrl+C path: main() catches KeyboardInterrupt, prints a message and exits 0.
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        return _fail(name, "did not shut down within 20s after SIGINT")

    t.join(timeout=5)
    out = "".join(captured)
    clean = ("Received Ctrl+C" in out) or (proc.returncode in (0, -signal.SIGINT, 130))

    if not http_ok:
        return _fail(name, f"served URL not reachable: {url} ({http_detail})")
    if not clean:
        return _fail(name, f"unclean shutdown (rc={proc.returncode})\n" + out[-800:])
    return _ok(name, f"served {url}; clean SIGINT shutdown (rc={proc.returncode})")


CHECKS = {
    "writer": check_writer,
    "launch": check_launch,
    "cli": check_cli,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", required=True, choices=sorted(CHECKS),
                        help="Which headless component check to run.")
    parser.add_argument("--workdir", required=True,
                        help="Scratch directory for log files.")
    args = parser.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    print(f"display_metrics headless check: {args.check}")
    ok = CHECKS[args.check](args.workdir)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
