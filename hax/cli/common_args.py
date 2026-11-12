"""Parametrized argparse builders for arguments shared across hax programs.

Many hax programs (HetSIREN, Zernike3Deep, ReconSIREN, the gray-scale adjusters,
MoDART, ...) expose the same data/training arguments. These helpers centralize
the canonical *structure* (flag name, type, action, choices, required-ness,
validation) and a canonical help string, while letting each program override the
parts that legitimately differ (e.g. ``--batch_size`` default, ``--reload`` help
text, ``--mode`` choices). Programs call e.g. ``add_sr(parser)`` instead of
repeating ``parser.add_argument("--sr", ...)``.

Design notes:
- ``add_*`` builders mirror ``ArgumentParser.add_argument`` semantics and return
  the created action, so a caller can tweak it further if ever needed.
- Help strings reuse the ``bcolors`` ANSI codes, exactly like the inline help
  they replace, so the rendered ``--help`` output (and the GUI form-schema
  introspected from it) is unchanged.
"""

import os
import re
import sys
import json
import argparse

from hax.utils import bcolors


def _list_arg_items(arg):
    """Split a list-valued CLI string tolerantly.

    Accepts the plain comma form (``0.8,0.2``) as well as the ways a list value
    realistically reaches argparse second-hand: a Python/YAML repr pasted back
    verbatim (``(0.25, 0.5, 0.75)``, ``[0.25, 0.5]``) - e.g. a GUI form echoing
    a tuple default, or a hand-written config - plus stray whitespace,
    whitespace-only separation and trailing commas.
    """
    text = str(arg).strip()
    pairs = {"(": ")", "[": "]", "{": "}"}
    if text and text[0] in pairs and text.endswith(pairs[text[0]]):
        text = text[1:-1]
    return [item for item in re.split(r"[\s,]+", text.strip()) if item]


def list_of_floats(arg):
    """argparse ``type`` for a list of floats (e.g. ``0.8,0.2`` or ``(0.8, 0.2)``)."""
    if isinstance(arg, (list, tuple)):
        return [float(item) for item in arg]
    return [float(item) for item in _list_arg_items(arg)]


def list_of_ints(arg):
    """argparse ``type`` for a list of ints (e.g. ``64,128`` or ``(64, 128)``)."""
    if isinstance(arg, (list, tuple)):
        return [int(item) for item in arg]
    return [int(item) for item in _list_arg_items(arg)]


def batch_size_or_auto(arg):
    """argparse ``type`` for ``--batch_size``: a positive int, or the string ``auto``.

    ``auto`` is a sentinel a program can resolve at run time (e.g. via
    :func:`hax.utils.estimate_batch_size`) into the largest memory-safe batch for
    the user's GPU. Everything else must parse as an integer >= 1.
    """
    if isinstance(arg, str) and arg.strip().lower() == "auto":
        return "auto"
    try:
        value = int(arg)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"--batch_size must be a positive integer or 'auto' (got {arg!r})")
    if value < 1:
        raise argparse.ArgumentTypeError(f"--batch_size must be >= 1 (got {value})")
    return value


# --------------------------------------------------------------------------- #
# Canonical help strings (the variant shared by most programs). A program with a
# legitimately different description passes its own ``help=`` to the builder.
# --------------------------------------------------------------------------- #

MD_HELP = "Xmipp/Relion metadata file with the images (+ alignments / CTF) to be analyzed"

SR_HELP = "Sampling rate of the images/volume"

CTF_TYPE_CHOICES = ["None", "apply", "wiener", "precorrect"]
CTF_TYPE_CHOICES_PREMULTIPLIED = CTF_TYPE_CHOICES + ["premultiplied"]
CTF_TYPE_HELP = ("Determines whether to consider the CTF and, in case it is considered, whether it will be "
                 "applied to the projections (apply) or used to correct the metadata images (wiener - precorrect)")

LOAD_IMAGES_TO_RAM_HELP = (
    f"If provided, images will be loaded to RAM. This is recommended if you want the best performance and your dataset fits in your RAM memory. If this flag is not provided, "
    f"images will be memory mapped. When this happens, the program will trade disk space for performance. Thus, during the execution additional disk space will be used and the performance "
    f"will be slightly lower compared to loading the images to RAM. Disk usage will be back to normal once the execution has finished.")

SSD_SCRATCH_FOLDER_HELP = (
    f"When the parameter {bcolors.UNDERLINE}load_images_to_ram{bcolors.ENDC} is not provided, we strongly recommend to provide here a path to a folder in a SSD disk to read faster the data. If not given, the data will be loaded from "
    f"the default disk.")

MODE_CHOICES = ["train", "predict"]
# Two-line variant (HetSIREN / ReconSIREN / FlexConsensus)
MODE_HELP = (f"{bcolors.BOLD}train{bcolors.ENDC}: train a neural network from scratch or from a previous execution if reload is provided\n"
             f"{bcolors.BOLD}predict{bcolors.ENDC}: predict the latent vectors from the input images ({bcolors.UNDERLINE}reload{bcolors.ENDC} parameter is mandatory in this case)")

EPOCHS_HELP = ("Number of epochs to train the network (i.e. how many times to loop over the whole dataset of images - set to default to 50 - "
               "as a rule of thumb, consider 50 to 100 epochs enough for 100k images / if your dataset is bigger or smaller, scale this value proportionally to it")

BATCH_SIZE_HELP = ("Determines how many images will be load in the GPU at any moment during training (set by default to 8 - "
                   f"you can control GPU memory usage easily by tuning this parameter to fit your hardware requirements - we recommend using tools like {bcolors.UNDERLINE}nvidia-smi{bcolors.ENDC} "
                   f"to monitor and/or measure memory usage and adjust this value - keep also in mind that bigger batch sizes might be less precise when looking for very local motions - "
                   f"pass {bcolors.ITALIC}auto{bcolors.ENDC} to let hax estimate the largest batch size that fits on your GPU (this maximizes GPU utilization, not necessarily accuracy)")
# Shorter variant used by the gray-scale adjusters / covariance / deconvolution programs.
BATCH_SIZE_HELP_ADJUST = ("Determines how many images will be load in the GPU at any moment during training (set by default to 8 - "
                          f"you can control GPU memory usage easily by tuning this parameter to fit your hardware requirements - we recommend using tools like {bcolors.UNDERLINE}nvidia-smi{bcolors.ENDC} "
                          f"to monitor and/or measure memory usage and adjust this value")

LEARNING_RATE_HELP = (
    f"The learning rate ({bcolors.ITALIC}lr{bcolors.ENDC}) sets the speed of learning. Think of the model as trying to find the lowest point in a valley; the {bcolors.ITALIC}lr{bcolors.ENDC} "
    f"is the size of the step it takes on each attempt. A large {bcolors.ITALIC}lr{bcolors.ENDC} (e.g., {bcolors.ITALIC}0.01{bcolors.ENDC}) is like taking huge leaps — it's fast but can be unstable, "
    f"overshoot the lowest point, or cause {bcolors.ITALIC}NAN{bcolors.ENDC} errors. A small {bcolors.ITALIC}lr{bcolors.ENDC} (e.g., {bcolors.ITALIC}1e-6{bcolors.ENDC}) is like taking tiny "
    f"shuffles — it's stable but very slow and might get stuck before reaching the bottom. A good default is often {bcolors.ITALIC}0.0001{bcolors.ENDC}. If training fails or errors explode, "
    f"try making the {bcolors.ITALIC}lr{bcolors.ENDC} 10 times smaller (e.g., {bcolors.ITALIC}0.001{bcolors.ENDC} --> {bcolors.ITALIC}0.0001{bcolors.ENDC}).")

DATASET_SPLIT_FRACTION_HELP = (
    f"Here you can provide the fractions to split your data automatically into a training and a validation subset following the format: {bcolors.ITALIC}training_fraction{bcolors.ENDC},"
    f"{bcolors.ITALIC}validation_fraction{bcolors.ENDC}. While the training subset will be used to train/update the network parameters, the validation subset will only be used to evaluate the "
    f"accuracy of the network when faced with new data. Therefore, the validation subset will never be used to update the networks parameters. {bcolors.WARNING}NOTE{bcolors.ENDC}: the sum of "
    f"{bcolors.ITALIC}training_fraction{bcolors.ENDC} and {bcolors.ITALIC}validation_fraction{bcolors.ENDC} must be equal to one.")

OUTPUT_PATH_HELP = "Path to save the results (trained neural network, new metadata...)"

LAT_DIM_HELP = "Dimensionality of the latent space of the network (set by default to 8)"

# Basic reload help (no HetSIREN/Zernike3Deep gray-level note), shared by the
# gray-scale adjusters, deconvolution and consensus programs.
RELOAD_HELP_BASIC = "Path to a folder containing an already saved neural network (useful to fine tune a previous network - predict from new data)"

SYMMETRY_GROUP_HELP = (
    f"If your protein has any kind of symmetry, you may pass it here so that it is considered while learning the angular assignment and the volume ({bcolors.WARNING}NOTE{bcolors.ENDC}: "
    f"only {bcolors.ITALIC}c*{bcolors.ENDC} and {bcolors.ITALIC}d*{bcolors.ENDC} symmetry groups are currently supported - the parameter is lower case sensitive - even if symmetry is provided, "
    f"the network will learn a {bcolors.ITALIC}symmetry broken{bcolors.ENDC} set of angles in c1. Therefore, the angles can be directly used in a reconstruction/refinement.)")

LOG_IMAGES_EVERY_HELP = (
    f"How often (in epochs) to log the {bcolors.ITALIC}cheap{bcolors.ENDC} intermediate results to Tensorboard: the predicted images and the central slices of the "
    f"predicted volumes (set by default to 1, i.e. every epoch - set to {bcolors.ITALIC}0{bcolors.ENDC} to disable)")

LOG_LANDSCAPE_EVERY_HELP = (
    f"How often (in epochs) to log the {bcolors.ITALIC}expensive{bcolors.ENDC} intermediate results: the latent space embedding (Tensorboard projector), the intermediate "
    f"volumes written to disk and the angular distribution plots (set by default to 5 - set to {bcolors.ITALIC}0{bcolors.ENDC} to disable). {bcolors.WARNING}NOTE{bcolors.ENDC}: the "
    f"embedding alone costs several seconds per call, so on small datasets it can easily dominate the training time - raise this value (or disable it) if logging is your bottleneck")

LOG_CHECKPOINT_EVERY_HELP = (
    f"How often (in epochs) to write the intermediate checkpoint used to resume an interrupted training (set by default to 5 - set to {bcolors.ITALIC}0{bcolors.ENDC} to disable). "
    f"This is deliberately independent from {bcolors.ITALIC}--log_landscape_every{bcolors.ENDC}: how often you can resume should not be tied to how often you want pictures")

LOG_TIME_BUDGET_HELP = (
    f"Self-tuning guard: the maximum fraction of the total wall clock that may be spent logging (e.g. {bcolors.ITALIC}0.05{bcolors.ENDC} for 5%%). When the logging carried out so far "
    f"exceeds this share of the run, the expensive tiers are skipped until training catches up. This adapts to dataset and box size on its own, which a fixed number of epochs "
    f"cannot do (set by default to 0, i.e. disabled - the cadence flags alone decide)")

LOG_SYNC_HELP = (
    f"Log on the training thread instead of on a background one. By default the host-side logging work (Tensorboard writes, {bcolors.ITALIC}.mrc{bcolors.ENDC} files, plots) is "
    f"moved to a background thread so that it overlaps with the next epoch's GPU work rather than stalling it. Use this flag if you need the logs to be written in lockstep with "
    f"training (e.g. while debugging)")


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #

def add_md(parser, required=True, help=MD_HELP):
    return parser.add_argument("--md", required=required, type=str, help=help)


def add_vol(parser, required=False, help=None):
    return parser.add_argument("--vol", required=required, type=str, help=help)


def add_mask(parser, required=False, help=None):
    return parser.add_argument("--mask", required=required, type=str, help=help)


def add_sr(parser, required=True, help=SR_HELP):
    return parser.add_argument("--sr", required=required, type=float, help=help)


def add_ctf_type(parser, required=True, choices=None, help=CTF_TYPE_HELP):
    return parser.add_argument("--ctf_type", required=required, type=str,
                               choices=choices if choices is not None else CTF_TYPE_CHOICES,
                               help=help)


def add_mode(parser, required=True, choices=None, help=MODE_HELP):
    return parser.add_argument("--mode", required=required, type=str,
                               choices=choices if choices is not None else MODE_CHOICES,
                               help=help)


def add_load_images_to_ram(parser, help=LOAD_IMAGES_TO_RAM_HELP):
    return parser.add_argument("--load_images_to_ram", action='store_true', help=help)


def add_ssd_scratch_folder(parser, help=SSD_SCRATCH_FOLDER_HELP):
    return parser.add_argument("--ssd_scratch_folder", required=False, type=str, help=help)


def add_symmetry_group(parser, default="c1", help=SYMMETRY_GROUP_HELP):
    return parser.add_argument("--symmetry_group", type=str, default=default, help=help)


def add_lat_dim(parser, default=8, help=LAT_DIM_HELP):
    return parser.add_argument("--lat_dim", required=False, type=int, default=default, help=help)


def add_epochs(parser, default=50, help=EPOCHS_HELP):
    return parser.add_argument("--epochs", required=False, type=int, default=default, help=help)


def add_batch_size(parser, default=8, help=BATCH_SIZE_HELP):
    return parser.add_argument("--batch_size", required=False, type=batch_size_or_auto, default=default, help=help)


def add_learning_rate(parser, default=1e-4, help=LEARNING_RATE_HELP):
    return parser.add_argument("--learning_rate", required=False, type=float, default=default, help=help)


def add_dataset_split_fraction(parser, default=(0.8, 0.2), help=DATASET_SPLIT_FRACTION_HELP):
    return parser.add_argument("--dataset_split_fraction", required=False,
                               type=list_of_floats, default=list(default), help=help)


def add_output_path(parser, required=True, help=OUTPUT_PATH_HELP):
    return parser.add_argument("--output_path", required=required, type=str, help=help)


def add_reload(parser, required=False, help=None):
    return parser.add_argument("--reload", required=required, type=str, help=help)


def add_logging_args(parser, image_every=1, landscape_every=5, checkpoint_every=5,
                     images=True, landscape=True, checkpoint=True):
    """Cadence of the intermediate logging, shared by the training programs.

    The tiers differ in cost by orders of magnitude (a loss scalar is free, volume
    slices are ~0.07 s, a latent embedding is several seconds), so each gets its own
    period instead of a single global one. See ``hax.metrics.TrainingLogger``.
    """
    if images:
        parser.add_argument("--log_images_every", required=False, type=int, default=image_every,
                            help=LOG_IMAGES_EVERY_HELP)
    if landscape:
        parser.add_argument("--log_landscape_every", required=False, type=int, default=landscape_every,
                            help=LOG_LANDSCAPE_EVERY_HELP)
    if checkpoint:
        parser.add_argument("--log_checkpoint_every", required=False, type=int, default=checkpoint_every,
                            help=LOG_CHECKPOINT_EVERY_HELP)
    parser.add_argument("--log_time_budget", required=False, type=float, default=0.0,
                        help=LOG_TIME_BUDGET_HELP)
    parser.add_argument("--log_sync", action='store_true', help=LOG_SYNC_HELP)
    return parser


def validate_dataset_split_fraction(fractions):
    """Exit cleanly (no traceback) if the train/validation split does not sum to one.

    Shared check used by the training programs. Emits a one-line message on
    stderr and exits with code 2 (the argparse convention for a usage error),
    instead of raising a ValueError that would surface as a Python traceback.
    """
    if sum(fractions) != 1:
        print(
            f"error: --dataset_split_fraction: the sum of {bcolors.ITALIC}training_fraction{bcolors.ENDC} and "
            f"{bcolors.ITALIC}validation_fraction{bcolors.ENDC} must equal 1 (got {fractions}).",
            file=sys.stderr)
        raise SystemExit(2)


# --------------------------------------------------------------------------- #
# Config-file support (Phase 3)
#
# A program can replace ``args = parser.parse_args()`` with
# ``args = common_args.parse_with_config(parser)`` to gain:
#   * a ``--config <file.yaml|.json>`` argument whose values are layered UNDER
#     the command line (explicit CLI args always win), and
#   * an automatic dump of the fully-resolved parameters to
#     ``<output_path>/run_config.json`` for reproducibility.
# --------------------------------------------------------------------------- #

CONFIG_HELP = (
    "Path to a YAML/JSON file with parameter values (keys are the argument names without the leading '--'). "
    f"Values from the file act as defaults; any argument also given on the command line overrides them. "
    f"{bcolors.WARNING}NOTE{bcolors.ENDC}: the fully-resolved parameters (file + command line) are written to "
    f"{bcolors.UNDERLINE}<output_path>/run_config.json{bcolors.ENDC} on every run, and that file can be reused "
    f"as a {bcolors.ITALIC}--config{bcolors.ENDC} to reproduce the run.")


def add_config(parser, help=CONFIG_HELP):
    return parser.add_argument("--config", required=False, type=str, help=help)


def _peek_config(argv):
    """Find a ``--config`` value in argv without invoking argparse.

    We must not call any ``ArgumentParser.parse_*`` here: the GUI introspector
    monkeypatches those to capture the parser, so a parse call for the peek would
    capture the wrong (or a throwaway) parser. A manual scan keeps the real
    ``parser.parse_args()`` below as the first/only parse call.
    """
    for i, tok in enumerate(argv):
        if tok == "--config" and i + 1 < len(argv):
            return argv[i + 1]
        if tok.startswith("--config="):
            return tok.split("=", 1)[1]
    return None


def _load_config_file(path):
    ext = os.path.splitext(path)[1].lower()
    with open(path) as fh:
        if ext in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError:
                raise SystemExit("PyYAML is required to read YAML config files; install pyyaml or use a .json config.")
            data = yaml.safe_load(fh)
        elif ext == ".json":
            data = json.load(fh)
        else:
            raise SystemExit(f"Unsupported config extension '{ext}'. Use .yaml, .yml or .json.")
    if not isinstance(data, dict):
        raise SystemExit("Config file must contain a top-level mapping of argument names to values.")
    return data


def save_run_config(args, filename="run_config.json"):
    """Best-effort dump of the resolved parameters into ``output_path``."""
    output_path = getattr(args, "output_path", None)
    if not output_path:
        return
    try:
        os.makedirs(output_path, exist_ok=True)
        resolved = {k: v for k, v in vars(args).items() if k != "config"}
        with open(os.path.join(output_path, filename), "w") as fh:
            json.dump(resolved, fh, indent=2, sort_keys=True, default=str)
    except OSError:
        # Provenance is a nice-to-have; never fail a run because it couldn't be written.
        pass


def parse_with_config(parser, argv=None):
    """``parser.parse_args()`` with optional ``--config`` layering + run dump.

    Precedence: command-line args > config-file values > argparse defaults.
    Config values can also satisfy ``required=True`` arguments (their
    requiredness is relaxed when the key is present in the file).
    """
    if not any("--config" in a.option_strings for a in parser._actions):
        add_config(parser)

    argv = sys.argv[1:] if argv is None else argv
    config_path = _peek_config(argv)
    if config_path:
        data = _load_config_file(config_path)
        known = {a.dest for a in parser._actions}
        unknown = sorted(set(data) - known)
        if unknown:
            raise SystemExit(f"Unknown keys in config file '{config_path}': {unknown}")
        for action in parser._actions:
            if action.dest in data:
                action.required = False
        parser.set_defaults(**data)

    args = parser.parse_args()
    save_run_config(args)
    return args
