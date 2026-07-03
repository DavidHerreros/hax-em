import os
import sys
import argparse
import difflib
import importlib
from importlib.metadata import version, PackageNotFoundError


# NOTE: the dispatcher sets CUDA_VISIBLE_DEVICES / XLA flags / the multiprocessing
# start method in main() *before the selected program is imported*. JAX reads
# those at backend initialization (first device use), not at ``import jax``, so
# this ordering is what matters in practice.
#
# We keep a local copy of the ANSI codes instead of ``from hax.utils import
# bcolors`` to avoid coupling the dispatcher to the heavy package for trivial
# constants. (Note: importing the program module below still runs
# ``hax/__init__.py`` -> ``from .networks import *``, which imports JAX. Making
# the entry point fully JAX-free would require a lazy ``hax/__init__.py`` and is
# tracked as a separate refactor.)
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


# Mirror hax.utils.loggers: emit ANSI codes only on a real terminal, honoring
# NO_COLOR / FORCE_COLOR. Kept local so the dispatcher stays free of heavy imports.
if os.environ.get("NO_COLOR") or (not os.environ.get("FORCE_COLOR") and not sys.stdout.isatty()):
    for _name in ("HEADER", "OKBLUE", "OKCYAN", "OKGREEN", "WARNING",
                  "FAIL", "ENDC", "BOLD", "UNDERLINE", "ITALIC"):
        setattr(bcolors, _name, "")


MODULES_DICT = {
    "hetsiren": ("hax.networks.hetsiren", "Heterogeneous volume reconstruction with HetSIREN neural network"),
    "zernike3deep": ("hax.networks.zernike3deep", "Estimation of motions using deep learning version of Zernike3Deep"),
    "flexconsensus": ("hax.networks.flexconsensus", "Consensus of conformational latent spaces using FlexConsensus neural network"),
    "latent_space_deconvolution": ("hax.networks.latent_space_deconvolution", "Deconvolution of conformational latent spaces"),
    "image_gray_scale_adjustment": ("hax.networks.image_gray_scale_adjustment", "Adjustment of volume projections to match a set of images"),
    "volume_gray_scale_adjustment": ("hax.networks.volume_gray_scale_adjustment", "Volume gray level adjustment towards a set of images"),
    "estimate_latent_covariances": ("hax.programs.estimate_latent_covariances", "Estimate latent space covariances matrices by simulating experimental images - needed by latent_space_deconvolution"),
    "decode_states_from_latents": ("hax.programs.decode_states_from_latents", "Decode a set of volumes given a network (Zernike3D or HetSIREN) and a set of latent vectors"),
    "filter_latents": ("hax.programs.filter_latents", "Filtering of latent spaces based on z-scores"),
    "display_metrics": ("hax.metrics.writer", "Display the model metrics (training curves, validation curves...) extracted while training a neural network"),
    "annotate_space": ("hax.viewers.annotate_space.annotate_space", "Interactive latent space analysis with real time map generation"),
    "reconsiren": ("hax.networks.reconsiren", "Ab initio estimation of particle pose, shifts and initial volume with neural networks"),
    "reconsiren_het_only": ("hax.networks.reconsiren_het_only", "Ab initio estimation of particle pose, shifts and initial volume with neural networks"),
    "modart": ("hax.programs.modart", "ART based volume reconstruction with motion correction to motion blurr artifacts"),
    "update": ("hax.cli.updater", "Check for and apply hax updates (PyPI release, or new commits for a devel/git install)")
}


def _get_version():
    try:
        return version("hax-em")
    except PackageNotFoundError:
        return "unknown"


class PrintSummary(argparse.Action):
    def __init__(self, option_strings, dest=argparse.SUPPRESS,
                 default=argparse.SUPPRESS, help=None):
        super().__init__(
            option_strings=option_strings,
            dest=dest,
            default=default,
            nargs=0,
            help=help
        )

    def __call__(self, parser, namespace, values, option_string=None):
        print(f"\n{bcolors.HEADER}SUMMARY OF AVAILABLE PROGRAMS{bcolors.ENDC}\n")
        for key, value in MODULES_DICT.items():
            print(f"     - {bcolors.ITALIC}{bcolors.BOLD}{key}{bcolors.ENDC}: {value[1]}")

        print(f"\n{bcolors.HEADER}Example of usage:{bcolors.ENDC}\n")
        print("     hax_project_manager --gpu 0 {Run only on this GPU} program --program_arg_1 #Val_1 --program arg_2 #Val_2 ...\n")
        print(f"{bcolors.HEADER}Additional help on how to execute each is available through:{bcolors.ENDC}\n")
        print(f"     hax_project_manager {bcolors.ITALIC}{bcolors.BOLD}program{bcolors.ENDC} {{-h or --help}}\n")
        print(f"{bcolors.WARNING}If you experience any issue or have suggestions, you are welcome to write an issue in our GitHub!: {bcolors.UNDERLINE}https://github.com/DavidHerreros/hax-em/issues\n{bcolors.ENDC}\n")
        print(f"{bcolors.OKBLUE}Documentation and tutorials on how to use the software (also with Scipion) are available at: {bcolors.UNDERLINE}https://davidherreros.github.io/hax-em-docs/\n{bcolors.ENDC}\n")
        parser.exit(0)


def _configure_environment(gpu):
    """Set environment variables that JAX/XLA read at backend initialization.

    Must run before the selected program (and therefore JAX) is imported.
    """
    # 1) set the GPU visibility before any JAX import
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ.pop("LD_LIBRARY_PATH", None)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_FLAGS"] = (
        "--xla_gpu_triton_gemm_any=true "
        "--xla_gpu_enable_latency_hiding_scheduler=true "
        "--xla_gpu_enable_highest_priority_async_stream=true "
    )
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def _quiet_dependency_logging():
    """Pin noisy third-party loggers to WARNING so their INFO chatter stays hidden.

    hax itself never configures logging, so historically these programs were
    quiet: with no handler on the root logger, Python's last-resort handler only
    shows WARNING and above. Two things break that assumption at runtime:

    * ``absl.logging`` (used internally by ``grain``/``orbax``) auto-attaches a
      handler to the root logger the first time it logs -- see the
      ``if not logging.root.handlers: logging.basicConfig()`` fallback in
      ``absl/logging/__init__.py``. Once a root handler exists, INFO records are
      no longer swallowed.
    * A host process (e.g. Scipion, whose ``rich`` handler produces the
      ``[MM/DD/YY HH:MM:SS] INFO`` lines) may set the root logger to INFO.

    Either way we start leaking dependency INFO lines such as JAX's
    "Unable to initialize backend 'tpu'" backend probe and grain's
    "Creating BatchOperation to enable SharedMemoryArray." Setting the level on
    the *emitting* loggers gates those records at the source, regardless of which
    handler is attached to the root. Set ``HAX_KEEP_DEP_LOGS`` to keep them.
    """
    if os.environ.get("HAX_KEEP_DEP_LOGS"):
        return
    import logging
    for name in ("jax", "absl", "grain", "orbax", "orbax.checkpoint",
                 "tensorboard", "tensorboardX"):
        logging.getLogger(name).setLevel(logging.WARNING)


def _configure_multiprocessing():
    """Set the default multiprocessing start method to ``forkserver``.

    Plain ``fork`` (the Linux default) is unsafe once CUDA/JAX has been
    initialized in the parent process, so we prefer ``forkserver``: workers are
    forked from a clean server process instead of the CUDA-loaded main process.
    This is set here (before the program and JAX are imported) rather than inside
    an ``if __name__ == "__main__"`` block, because the installed entry point is
    ``hax.cli:main`` and that block never runs under the console script.
    ``force=True`` makes the call idempotent; we guard against platforms where
    ``forkserver`` is unavailable.
    """
    import multiprocessing as mp
    try:
        mp.set_start_method("forkserver", force=True)
    except (ValueError, RuntimeError):
        # forkserver not available on this platform; keep the default method.
        pass


def _resolve_program(parser, program):
    """Map a CLI program name to its module path, or fail with a helpful hint."""
    if program not in MODULES_DICT:
        suggestions = difflib.get_close_matches(program, list(MODULES_DICT), n=3)
        message = f"{bcolors.FAIL}Unknown program '{program}'.{bcolors.ENDC}\n"
        if suggestions:
            message += f"Did you mean: {bcolors.BOLD}{', '.join(suggestions)}{bcolors.ENDC}?\n"
        message += (f"Run {bcolors.BOLD}hax_project_manager -h{bcolors.ENDC} "
                    f"to see the list of available programs.\n")
        parser.exit(2, message)
    return MODULES_DICT[program][0]


def main():
    parser = argparse.ArgumentParser(
        description="Command line interface to launch Hax programs",
        add_help=False
    )
    parser.add_argument(
        "--gpu", required=False,
        help="Which GPU(s) to expose (value for CUDA_VISIBLE_DEVICES)"
    )
    parser.add_argument(
        "-h", "--help", action=PrintSummary, default=argparse.SUPPRESS,
        help="Shows a summary of available commands"
    )
    parser.add_argument(
        "--version", action="version", version=f"hax-em {_get_version()}",
        help="Show the installed hax-em version and exit"
    )
    parser.add_argument(
        "program",
        help="The program to be executed"
    )
    parser.add_argument(
        "args", nargs=argparse.REMAINDER,
        help="Arguments to pass along to the previously selected program"
    )

    ns, _ = parser.parse_known_args()

    # 1) configure the environment and multiprocessing before importing JAX
    _configure_environment(ns.gpu)
    _configure_multiprocessing()
    _quiet_dependency_logging()

    # 2) resolve the program (friendly error on a bad name)
    module_path = _resolve_program(parser, ns.program)

    # 3) hand a clean argv to the program so it can parse its own arguments
    #    strictly (without seeing the dispatcher's --gpu/program prefix).
    sys.argv = [f"hax_project_manager {ns.program}"] + ns.args

    # 4) import the program module (this is what pulls in JAX) and run it
    module = importlib.import_module(module_path)
    main_fn = getattr(module, "main")
    main_fn()


if __name__ == "__main__":
    main()
