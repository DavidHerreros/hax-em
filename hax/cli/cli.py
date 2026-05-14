import os
import sys
import argparse
import subprocess
import importlib
from hax.utils import bcolors

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
    "modart": ("hax.programs.modart", "ART based volume reconstruction with motion correction to motion blurr artifacts")
}


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
        print(f"{bcolors.WARNING}If you experience any issue or have suggestions, you are welcome to write an issue in our GitHub!: {bcolors.UNDERLINE}https://github.com/DavidHerreros/Hax/issues\n{bcolors.ENDC}\n")
        print(f"{bcolors.OKBLUE}We also provide tutorials on how to use the software with Scipion in the following link: {bcolors.UNDERLINE}VERY SOON!\n{bcolors.ENDC}\n")
        parser.exit(0)

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
        "program",
        help="The program to be executed"
    )
    parser.add_argument(
        "args", nargs=argparse.REMAINDER,
        help="Arguments to pass along to the previously selected program"
    )

    ns, _ = parser.parse_known_args()

    # 1) set the env var before any JAX import
    if ns.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ns.gpu
    os.environ.pop("LD_LIBRARY_PATH", None)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_FLAGS"] = (
        "--xla_gpu_triton_gemm_any=true "
        "--xla_gpu_enable_latency_hiding_scheduler=true "
        "--xla_gpu_enable_highest_priority_async_stream=true "
    )
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    # 2) Prepare command
    module = importlib.import_module(MODULES_DICT[ns.program][0])
    main_fn = getattr(module, "main")

    # 3) Run function
    main_fn()

if __name__ == "__main__":
    import multiprocessing
    # multiprocessing.set_start_method("spawn", force=True)
    multiprocessing.set_start_method('forkserver', force=True)

    main()
