"""``hax_project_manager update`` — check for and apply hax-em updates.

Two installation scenarios are handled automatically:

* **Production (PyPI)** — installed from a released wheel. The command compares
  the installed version against the latest on PyPI and, if newer, upgrades with
  ``pip install -U "hax-em[<cuda extra>]"`` (the CUDA extra is detected from the
  installed cupy wheel).
* **Development (editable / git)** — installed with ``pip install -e`` from a
  local clone, or ``pip install git+...``. The command checks the tracked branch
  for new commits. For an editable clone it ``git pull``s; if the pulled changes
  touch ``pyproject.toml`` (i.e. dependencies may have changed) it warns and
  either reinstalls the package deps (``-y`` / on confirmation) or prints the
  exact command for the user to run themselves.

This module is intentionally free of heavy imports (no JAX / hax internals) so it
stays fast and robust as a maintenance entry point.
"""

import os
import sys
import json
import shlex
import shutil
import tempfile
import argparse
import subprocess
from importlib import metadata, util
from urllib import request, error

try:
    from packaging.version import Version
except Exception:  # pragma: no cover - packaging is normally always present
    Version = None

DIST = "hax-em"
PKG = "hax"
PYPI_URL = f"https://pypi.org/pypi/{DIST}/json"


# --------------------------------------------------------------------------- #
# Tiny TTY-aware color helpers (kept local so this module pulls no hax/JAX deps)
# --------------------------------------------------------------------------- #
def _color_enabled():
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


_COLOR = _color_enabled()


def _c(text, code):
    return f"\033[{code}m{text}\033[0m" if _COLOR else text


def _bold(t): return _c(t, "1")
def _blue(t): return _c(t, "94")
def _green(t): return _c(t, "92")
def _yellow(t): return _c(t, "93")
def _red(t): return _c(t, "91")


# --------------------------------------------------------------------------- #
# Discovery helpers
# --------------------------------------------------------------------------- #
def _package_dir():
    """Locate the hax package directory WITHOUT importing it (no JAX)."""
    spec = util.find_spec(PKG)
    if spec and spec.origin:
        return os.path.dirname(spec.origin)
    return None


def _find_git_repo(start):
    path = os.path.abspath(start)
    while True:
        if os.path.isdir(os.path.join(path, ".git")):
            return path
        parent = os.path.dirname(path)
        if parent == path:
            return None
        path = parent


def _direct_url():
    try:
        text = metadata.distribution(DIST).read_text("direct_url.json")
        return json.loads(text) if text else None
    except Exception:
        return None


def _detect_cuda_extra():
    """Return 'cuda12' / 'cuda13' based on the installed cupy wheel, else None."""
    for extra, pkg in (("cuda12", "cupy-cuda12x"), ("cuda13", "cupy-cuda13x")):
        try:
            metadata.version(pkg)
            return extra
        except metadata.PackageNotFoundError:
            continue
    return None


def _install_target(extra):
    return f"{DIST}[{extra}]" if extra else DIST


def detect_install():
    """Classify the installation: 'git' (editable clone), 'vcs' (git+), or 'pypi'."""
    pkg_dir = _package_dir()
    repo = _find_git_repo(pkg_dir) if pkg_dir else None
    if repo and os.path.exists(os.path.join(repo, "pyproject.toml")):
        return {"kind": "git", "repo": repo}
    du = _direct_url() or {}
    if "vcs_info" in du:
        return {"kind": "vcs", "direct_url": du}
    return {"kind": "pypi"}


# --------------------------------------------------------------------------- #
# Subprocess / prompt helpers
# --------------------------------------------------------------------------- #
def _git(repo, *args):
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True)


def _show(cmd):
    """Shell-safe, copy-pasteable rendering of a command list (quotes brackets)."""
    return " ".join(shlex.quote(a) for a in cmd)


def _run_live(cmd, cwd=None):
    """Run a command with inherited stdout/stderr so the user sees progress."""
    print(_bold(f"\n$ {_show(cmd)}\n"))
    return subprocess.run(cmd, cwd=cwd).returncode


def _confirm(prompt, assume_yes):
    if assume_yes:
        return True
    if not sys.stdin.isatty():
        return False
    try:
        return input(f"{prompt} [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def _pip(*args):
    return [sys.executable, "-m", "pip", *args]


def _newer(latest, installed):
    if Version is not None:
        try:
            return Version(latest) > Version(installed)
        except Exception:
            pass
    return latest != installed


# --------------------------------------------------------------------------- #
# Scenario handlers — each returns a process exit code
# --------------------------------------------------------------------------- #
def _handle_pypi(installed, check, assume_yes):
    print(f"Install type: {_bold('PyPI release')}  (installed {_bold(installed)})")
    try:
        with request.urlopen(PYPI_URL, timeout=15) as resp:
            latest = json.load(resp)["info"]["version"]
    except (error.URLError, OSError, ValueError, KeyError) as exc:
        print(_red(f"Could not query PyPI for the latest version: {exc}"))
        return 1

    if not _newer(latest, installed):
        print(_green(f"hax-em is up to date (latest on PyPI is {latest})."))
        return 0

    print(_yellow(f"A new release is available: {installed} -> {_bold(latest)}"))
    extra = _detect_cuda_extra()
    target = _install_target(extra)
    cmd = _pip("install", "--upgrade", target)
    if check:
        print(f"To update, run:\n    {_bold(_show(cmd))}")
        return 0
    if not _confirm(f"Upgrade hax-em to {latest} now?", assume_yes):
        print(f"Skipped. To update yourself, run:\n    {_bold(_show(cmd))}")
        return 0
    return _run_live(cmd)


def _handle_git(repo, check, assume_yes):
    print(f"Install type: {_bold('development (editable git clone)')}")
    print(f"Repository:   {repo}")

    branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    print(f"Branch:       {_bold(branch)}")

    fetched = _git(repo, "fetch", "--quiet")
    if fetched.returncode != 0:
        print(_red(f"git fetch failed (offline?):\n{fetched.stderr.strip()}"))
        return 1

    upstream = _git(repo, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
    if upstream.returncode != 0:
        print(_yellow(f"Branch '{branch}' has no upstream to compare against. "
                      f"Set one with:\n    git -C {repo} branch --set-upstream-to=origin/{branch}"))
        return 1

    behind = _git(repo, "rev-list", "--count", "HEAD..@{u}").stdout.strip() or "0"
    ahead = _git(repo, "rev-list", "--count", "@{u}..HEAD").stdout.strip() or "0"
    if behind == "0":
        extra = "" if ahead == "0" else _yellow(f" (local is {ahead} commit(s) ahead)")
        print(_green(f"hax-em is up to date on '{branch}'.") + extra)
        return 0

    print(_yellow(f"{behind} new commit(s) available on '{branch}':"))
    log = _git(repo, "log", "--oneline", "-n", "10", "HEAD..@{u}").stdout.strip()
    print(log)

    if check:
        print(f"\nTo apply, run:\n    {_bold('hax_project_manager update')}")
        return 0

    old = _git(repo, "rev-parse", "HEAD").stdout.strip()
    if _run_live(["git", "-C", repo, "pull", "--ff-only"]) != 0:
        print(_red("git pull --ff-only failed (local branch may have diverged). "
                   "Resolve manually, then re-run."))
        return 1
    new = _git(repo, "rev-parse", "HEAD").stdout.strip()

    # Did the pulled changes touch pyproject.toml (i.e. dependencies)?
    changed = _git(repo, "diff", "--name-only", old, new).stdout.split()
    extra = _detect_cuda_extra()
    target = f".[{extra}]" if extra else "."
    reinstall = _pip("install", "-e", target)
    reinstall_str = f'cd {shlex.quote(repo)} && {_show(reinstall)}'

    if "pyproject.toml" not in changed:
        print(_green("\nDone. Code updated (editable install); no dependency changes detected."))
        return 0

    print(_yellow("\nThe update modified pyproject.toml — dependencies may have changed."))
    if _confirm("Update/installed packages now (pip install -e the new deps)?", assume_yes):
        return _run_live(reinstall, cwd=repo)
    print("Skipped dependency update. To finish it yourself, run:\n"
          f"    {_bold(reinstall_str)}")
    return 0


def _remote_pyproject_changed(url, ref, installed_commit, remote_commit):
    """Best-effort: did pyproject.toml change between installed_commit and remote_commit?

    Returns True / False, or None if undeterminable (remote unreachable, ref not a
    branch/tag, or the installed commit not reachable). Uses a blobless, no-checkout
    clone so only the two pyproject.toml blobs are fetched, not the whole history.
    """
    if not installed_commit or not ref or ref == "HEAD":
        return None
    tmp = tempfile.mkdtemp(prefix="hax_update_")
    try:
        clone = subprocess.run(
            ["git", "clone", "--quiet", "--filter=blob:none", "--no-checkout",
             "--single-branch", "--branch", ref, url, tmp],
            capture_output=True, text=True)
        if clone.returncode != 0:
            return None

        def _present(commit):
            return subprocess.run(["git", "-C", tmp, "cat-file", "-e", f"{commit}^{{commit}}"],
                                  capture_output=True).returncode == 0

        if not _present(installed_commit):
            subprocess.run(["git", "-C", tmp, "fetch", "--quiet", "origin", installed_commit],
                           capture_output=True)
        if not _present(installed_commit):
            return None
        diff = subprocess.run(
            ["git", "-C", tmp, "diff", "--name-only", installed_commit, remote_commit, "--", "pyproject.toml"],
            capture_output=True, text=True)
        if diff.returncode != 0:
            return None
        return bool(diff.stdout.strip())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _handle_vcs(direct_url, check, assume_yes):
    info = direct_url.get("vcs_info", {})
    url = direct_url.get("url", "")
    branch = info.get("requested_revision")
    installed_commit = info.get("commit_id", "")
    print(f"Install type: {_bold('development (git+ install)')}")
    print(f"Source:       {url}" + (f" @ {branch}" if branch else ""))

    ref = branch or "HEAD"
    ls = subprocess.run(["git", "ls-remote", url, ref], capture_output=True, text=True)
    if ls.returncode != 0 or not ls.stdout.strip():
        print(_red(f"Could not query the remote for '{ref}':\n{ls.stderr.strip()}"))
        return 1
    remote_commit = ls.stdout.split()[0]

    if installed_commit and remote_commit.startswith(installed_commit[:12]):
        print(_green(f"hax-em is up to date with {ref} ({remote_commit[:12]})."))
        return 0

    print(_yellow(f"A newer commit is available on '{ref}': "
                  f"{installed_commit[:12] or '?'} -> {_bold(remote_commit[:12])}"))

    # Mirror the editable flow: code-only reinstall when dependencies are
    # unchanged, full (dependency-resolving) reinstall when pyproject.toml changed.
    deps_changed = _remote_pyproject_changed(url, ref, installed_commit, remote_commit)
    extra = _detect_cuda_extra()
    spec = f"git+{url}@{branch}" if branch else f"git+{url}"
    # PEP 508 direct reference so the CUDA extra is applied reliably.
    full_target = f"{DIST}[{extra}] @ {spec}" if extra else spec
    full_cmd = _pip("install", "--upgrade", "--force-reinstall", full_target)
    code_cmd = _pip("install", "--upgrade", "--force-reinstall", "--no-deps", spec)

    if check:
        if deps_changed is False:
            print("Dependencies are unchanged — code-only update. To update, run:\n"
                  f"    {_bold(_show(code_cmd))}")
        else:
            note = "" if deps_changed else " (could not verify dependency changes)"
            print(f"This update may change dependencies{note}. To update, run:\n"
                  f"    {_bold(_show(full_cmd))}")
        return 0

    if deps_changed is False:
        # Safe, analogous to an editable `git pull`: refresh the code, leave deps.
        print("Dependencies unchanged; updating code only (--no-deps).")
        return _run_live(code_cmd)

    if deps_changed is None:
        print(_yellow("Could not determine whether dependencies changed; a full reinstall "
                      "(with dependency resolution) will be used."))
    else:
        print(_yellow("This update modifies pyproject.toml — dependencies will be updated/added."))
    if _confirm("Proceed with the full reinstall (updates packages too)?", assume_yes):
        return _run_live(full_cmd)
    print(f"Skipped. To finish it yourself, run:\n    {_bold(_show(full_cmd))}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        prog="hax_project_manager update",
        description="Check for and apply updates to hax-em (PyPI release or development git branch).")
    parser.add_argument("--check", action="store_true",
                        help="Only check for updates and report; do not modify anything.")
    parser.add_argument("-y", "--yes", action="store_true",
                        help="Assume 'yes' to all prompts and apply the full update automatically "
                             "(including dependency reinstall for development installs).")
    args = parser.parse_args()

    try:
        installed = metadata.version(DIST)
    except metadata.PackageNotFoundError:
        print(_red("hax-em does not appear to be installed in this environment."))
        sys.exit(1)

    info = detect_install()
    kind = info["kind"]
    if kind == "pypi":
        rc = _handle_pypi(installed, args.check, args.yes)
    elif kind == "git":
        rc = _handle_git(info["repo"], args.check, args.yes)
    else:  # vcs
        rc = _handle_vcs(info["direct_url"], args.check, args.yes)
    sys.exit(rc)


if __name__ == "__main__":
    main()
