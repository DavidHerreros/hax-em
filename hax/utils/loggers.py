import os
import sys


def _color_enabled():
    """Decide whether ANSI color codes should be emitted.

    Honors the de-facto conventions ``NO_COLOR`` (disable) and ``FORCE_COLOR``
    (force on); otherwise colors are emitted only when stdout is a real terminal.
    This keeps escape sequences out of piped output, redirected log files and
    captures (e.g. Scipion or the GUI launching programs as subprocesses), while
    preserving colored ``--help`` and messages in an interactive shell.
    """
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    ITALIC = '\033[3m'


# Blank out the codes (to empty strings) when color is disabled, so every
# ``bcolors.X`` reference across the codebase becomes a no-op transparently.
if not _color_enabled():
    for _name in ("HEADER", "OKBLUE", "OKCYAN", "OKGREEN", "WARNING",
                  "FAIL", "ENDC", "BOLD", "UNDERLINE", "ITALIC"):
        setattr(bcolors, _name, "")
