#!/usr/bin/env python
"""
Semantic verification for the preprocessing programs, run as a subprocess check.

The suite's default pass criterion -- exit 0 plus the expected files -- cannot see
whether ``preprocess_particles`` actually *rescaled the in-plane shifts*, updated
the sampling rate, or preserved gray levels. Those are exactly the things a future
refactor would break silently, so this script re-derives them from the outputs the
CLI scenarios already produced and asserts them.

It follows the ``annotate_space_headless.py`` pattern: one ``--check <name>``
per invocation, driven from ``test_preprocess_*.py`` through ``Scenario(script=...)``.

Checks
------
``particles``        geometry, sampling rate, shift rescaling, carried-over metadata,
                     an independent NumPy oracle for the Fourier resize, and batch /
                     device / worker invariance of the pixel data.
``particles_guards`` the CLI rejects invalid geometry and impossible requests.
``volumes``          geometry, sampling rate, gray-level (mean) preservation, an
                     independent oracle, centre preservation, and the output naming
                     the GUI's single-volume connector relies on.
``volumes_guards``   the CLI rejects a no-op run and a non-cubic Fourier resize.

Nothing here imports a Hax *program* module: the outputs are read back with
``xmipp_metadata`` and the reference values recomputed with plain NumPy, so the
checks are independent of the implementation under test.
"""

import argparse
import os
import subprocess
import sys

import numpy as np
from xmipp_metadata.image_handler import ImageHandler
from xmipp_metadata.metadata import XmippMetaData

GREEN, RED, END = "\033[92m", "\033[91m", "\033[0m"

# Reproduces the ``hax_project_manager`` console entry point (see common.py).
_CLI_BOOTSTRAP = "from hax.cli import main; main()"


class Checks:
    """Tiny assertion collector: prints one line per check, tracks failures."""

    def __init__(self):
        self.failed = 0

    def __call__(self, name, ok, detail=""):
        ok = bool(ok)
        tag = f"{GREEN}OK  {END}" if ok else f"{RED}FAIL{END}"
        print(f"    {tag} {name}" + (f" — {detail}" if detail else ""), flush=True)
        if not ok:
            self.failed += 1
        return ok

    def close(self):
        print(f"\n  {self.failed} failed check(s)" if self.failed else "\n  all checks passed")
        return 1 if self.failed else 0


# --------------------------------------------------------------------------- #
#  Independent reference implementations (plain NumPy, from the definition)     #
# --------------------------------------------------------------------------- #
def ref_centered_crop(array, new, axes):
    """Centre crop/pad anchored on index ``n // 2`` (the fftshift/box-centre convention)."""
    slices = [slice(None)] * array.ndim
    pads = [(0, 0)] * array.ndim
    for axis in axes:
        old = array.shape[axis]
        offset = old // 2 - new // 2
        if offset >= 0:
            slices[axis] = slice(offset, offset + new)
        else:
            pads[axis] = (-offset, new - old + offset)
    out = array[tuple(slices)]
    return np.pad(out, pads) if any(p != (0, 0) for p in pads) else out


def ref_fourier_resize(array, new, axes):
    """Fourier crop/pad + the amplitude rescaling that preserves gray levels."""
    old = np.prod([array.shape[a] for a in axes])
    spectrum = np.fft.fftshift(np.fft.fftn(array, axes=axes), axes=axes)
    spectrum = ref_centered_crop(spectrum, new, axes)
    out = np.real(np.fft.ifftn(np.fft.ifftshift(spectrum, axes=axes), axes=axes))
    return out * (new ** len(axes) / old)


def close(a, b, tol=1e-3):
    """Max abs difference, normalised by the reference's spread (scale-free)."""
    scale = float(np.std(b)) or 1.0
    return float(np.abs(a - b).max()) / scale <= tol, float(np.abs(a - b).max()) / scale


# --------------------------------------------------------------------------- #
#  Readers                                                                      #
# --------------------------------------------------------------------------- #
def read_stack(run_dir, name="preprocessed_particles.mrcs"):
    handler = ImageHandler(os.path.join(run_dir, name))
    return np.squeeze(handler.getData()), handler.getSamplingRate()


def read_md(run_dir, ext="xmd"):
    return XmippMetaData(os.path.join(run_dir, f"preprocessed_particles.{ext}"))


def read_vol(path):
    handler = ImageHandler(path)
    return np.squeeze(handler.getData()), handler.getSamplingRate()


def centre_of_mass(volume):
    """COM in voxels, relative to the box centre (index n // 2)."""
    grids = np.meshgrid(*[np.arange(s) - s // 2 for s in volume.shape], indexing="ij")
    total = volume.sum()
    return np.array([float((g * volume).sum() / total) for g in grids])


# --------------------------------------------------------------------------- #
#  particles                                                                    #
# --------------------------------------------------------------------------- #
def check_particles(args):
    c = Checks()
    runs, box, sr = args.runs, args.box, args.sr
    src_md = XmippMetaData(args.source_md)
    src_images = src_md.getMetaDataImage(np.arange(len(src_md)))
    src_shifts = src_md.getMetaDataColumns(["shiftX", "shiftY"]).astype(np.float64)
    src_angles = src_md.getMetaDataColumns(["angleRot", "angleTilt", "anglePsi"]).astype(np.float64)
    src_ctf = src_md.getMetaDataColumns(["ctfDefocusU", "ctfDefocusV", "ctfDefocusAngle"]).astype(np.float64)
    n = len(src_md)

    # (run name, box after crop, box after resize) -> expected sr / shift factor.
    geometry = [
        ("resize", box, 24),
        ("crop", 32, 32),
        ("crop_resize", 32, 16),
        ("upsample", box, 64),
        ("pad", 64, 64),
        ("ctf_only", box, box),
        ("wiener", box, 24),
        ("phase_flip", box, 24),
        ("wiener_eps", box, 24),
        ("batch_auto", box, 24),
        ("device_cpu", box, 24),
        ("workers", box, 24),
        ("relative", box, 24),
    ]

    print("\n  geometry, sampling rate and shift rescaling")
    for name, box_cropped, box_out in geometry:
        run = os.path.join(runs, name)
        stack, voxel = read_stack(run)
        md = read_md(run)
        sr_out = sr * box_cropped / box_out
        shift_scale = box_out / box_cropped

        c(f"{name}:shape", stack.shape == (n, box_out, box_out), f"{stack.shape}")
        c(f"{name}:sampling_rate", abs(voxel - sr_out) < 1e-4, f"{voxel:.4f} (expected {sr_out:.4f})")
        c(f"{name}:n_particles", len(md) == n, f"{len(md)}")

        shifts = md.getMetaDataColumns(["shiftX", "shiftY"]).astype(np.float64)
        ok, err = close(shifts, src_shifts * shift_scale, tol=1e-4)
        c(f"{name}:shifts_rescaled", ok, f"x{shift_scale:.4f}, max rel err {err:.2e}")

        # Angles and CTF describe the specimen, not the box: they must survive untouched.
        c(f"{name}:angles_preserved",
          np.allclose(md.getMetaDataColumns(["angleRot", "angleTilt", "anglePsi"]).astype(np.float64), src_angles))
        c(f"{name}:ctf_columns_preserved",
          np.allclose(md.getMetaDataColumns(["ctfDefocusU", "ctfDefocusV", "ctfDefocusAngle"]).astype(np.float64), src_ctf))
        c(f"{name}:finite", np.isfinite(stack).all())

    print("\n  pixel data vs an independent NumPy oracle")
    resized, _ = read_stack(os.path.join(runs, "resize"))
    oracle = ref_fourier_resize(src_images.astype(np.float64), 24, axes=(-2, -1))
    ok, err = close(resized, oracle)
    c("resize:matches_fourier_oracle", ok, f"max rel err {err:.2e}")

    cropped, _ = read_stack(os.path.join(runs, "crop"))
    oracle = ref_centered_crop(src_images.astype(np.float64), 32, axes=(-2, -1))
    ok, err = close(cropped, oracle)
    c("crop:matches_centred_crop_oracle", ok, f"max rel err {err:.2e}")

    both, _ = read_stack(os.path.join(runs, "crop_resize"))
    oracle = ref_fourier_resize(ref_centered_crop(src_images.astype(np.float64), 32, axes=(-2, -1)),
                                16, axes=(-2, -1))
    ok, err = close(both, oracle)
    c("crop_resize:crop_then_resize_order", ok, f"max rel err {err:.2e}")

    padded, _ = read_stack(os.path.join(runs, "pad"))
    border = np.concatenate([padded[:, :8, :].ravel(), padded[:, -8:, :].ravel()])
    c("pad:zero_border", np.abs(border).max() < 1e-6, f"max |border| {np.abs(border).max():.2e}")

    # Gray levels: Fourier resizing keeps the DC term, so the per-image mean is exact.
    upsampled, _ = read_stack(os.path.join(runs, "upsample"))
    ok, err = close(upsampled.mean(axis=(1, 2)), src_images.mean(axis=(1, 2)), tol=1e-2)
    c("upsample:mean_preserved", ok, f"max rel err {err:.2e}")

    print("\n  the batch/device/worker knobs must not change the result")
    for name in ("batch_auto", "device_cpu", "workers"):
        other, _ = read_stack(os.path.join(runs, name))
        ok, err = close(other, resized, tol=1e-3)
        c(f"{name}:identical_to_resize", ok, f"max rel err {err:.2e}")

    print("\n  CTF correction")
    wiener, _ = read_stack(os.path.join(runs, "wiener"))
    flip, _ = read_stack(os.path.join(runs, "phase_flip"))
    eps, _ = read_stack(os.path.join(runs, "wiener_eps"))
    c("wiener:changes_pixels", not np.allclose(wiener, resized))
    c("phase_flip:changes_pixels", not np.allclose(flip, resized))
    c("wiener:differs_from_phase_flip", not np.allclose(wiener, flip))
    c("wiener_epsilon:takes_effect", not np.allclose(eps, wiener))
    ctf_only, voxel = read_stack(os.path.join(runs, "ctf_only"))
    c("ctf_only:geometry_untouched", ctf_only.shape == (n, box, box) and abs(voxel - sr) < 1e-4)
    c("ctf_only:changes_pixels", not np.allclose(ctf_only, src_images))

    print("\n  metadata image references")
    abs_md = read_md(os.path.join(runs, "resize"))
    entry = str(abs_md.getMetadataItems(0, "image")[0])
    c("default:absolute_image_path", os.path.isabs(entry.split("@")[-1]), entry.split("@")[-1])
    # Absolute paths must resolve from an unrelated working directory.
    cwd = os.getcwd()
    os.chdir(os.path.expanduser("~"))
    try:
        c("default:reads_from_any_cwd", read_md(os.path.join(runs, "resize")).getMetaDataImage(0).shape == (24, 24))
    finally:
        os.chdir(cwd)

    rel_md = read_md(os.path.join(runs, "relative"))
    rel_entry = str(rel_md.getMetadataItems(0, "image")[0]).split("@")[-1]
    c("relative:basename_only", rel_entry == "preprocessed_particles.mrcs", rel_entry)
    os.chdir(os.path.join(runs, "relative"))
    try:
        c("relative:resolves_from_output_dir",
          XmippMetaData("preprocessed_particles.xmd").getMetaDataImage(0).shape == (24, 24))
    finally:
        os.chdir(cwd)

    print("\n  metadata format follows the input extension")
    star = os.path.join(runs, "star", "preprocessed_particles.star")
    c("star:written", os.path.isfile(star))
    c("star:no_xmd_written", not os.path.isfile(os.path.join(runs, "star", "preprocessed_particles.xmd")))
    if os.path.isfile(star):
        star_md = XmippMetaData(star)
        c("star:readable", len(star_md) == n and star_md.getMetaDataImage(0).shape == (24, 24))

    return c.close()


# --------------------------------------------------------------------------- #
#  volumes                                                                      #
# --------------------------------------------------------------------------- #
def check_volumes(args):
    c = Checks()
    runs, box, sr = args.runs, args.box, args.sr
    src, _ = read_vol(args.source_vol)
    stem = os.path.splitext(os.path.basename(args.source_vol))[0]

    def out(run, name=None):
        return os.path.join(runs, run, name or f"{stem}_preprocessed.mrc")

    print("\n  geometry and sampling rate")
    for name, box_cropped, box_out in [("resize", box, 24), ("crop", 32, 32),
                                       ("crop_resize", 32, 16), ("upsample", box, 64),
                                       ("header_sr", box, 24), ("workers", box, 24)]:
        vol, voxel = read_vol(out(name))
        sr_out = sr * box_cropped / box_out
        c(f"{name}:shape", vol.shape == (box_out,) * 3, f"{vol.shape}")
        c(f"{name}:sampling_rate", abs(voxel - sr_out) < 1e-4, f"{voxel:.4f} (expected {sr_out:.4f})")
        c(f"{name}:finite", np.isfinite(vol).all())

    print("\n  voxel data vs an independent NumPy oracle")
    resized, _ = read_vol(out("resize"))
    oracle = ref_fourier_resize(src.astype(np.float64), 24, axes=(0, 1, 2))
    ok, err = close(resized, oracle)
    c("resize:matches_fourier_oracle", ok, f"max rel err {err:.2e}")

    cropped, _ = read_vol(out("crop"))
    ok, err = close(cropped, ref_centered_crop(src.astype(np.float64), 32, axes=(0, 1, 2)))
    c("crop:matches_centred_crop_oracle", ok, f"max rel err {err:.2e}")

    both, _ = read_vol(out("crop_resize"))
    oracle = ref_fourier_resize(ref_centered_crop(src.astype(np.float64), 32, axes=(0, 1, 2)), 16, axes=(0, 1, 2))
    ok, err = close(both, oracle)
    c("crop_resize:crop_then_resize_order", ok, f"max rel err {err:.2e}")

    print("\n  density is preserved")
    # Fourier resizing keeps the DC term, so the mean density is exactly preserved.
    c("resize:mean_density_preserved",
      abs(resized.mean() - src.mean()) / abs(src.mean()) < 1e-4,
      f"{src.mean():.5f} -> {resized.mean():.5f}")
    upsampled, _ = read_vol(out("upsample"))
    c("upsample:mean_density_preserved",
      abs(upsampled.mean() - src.mean()) / abs(src.mean()) < 1e-4,
      f"{src.mean():.5f} -> {upsampled.mean():.5f}")
    c("resize:peak_preserved", abs(resized.max() - src.max()) / src.max() < 0.05,
      f"{src.max():.4f} -> {resized.max():.4f}")

    # A centred resize must not move the molecule: the COM, measured from the box
    # centre, scales with the box and nothing else.
    com_src, com_new = centre_of_mass(src), centre_of_mass(resized)
    ok, err = close(com_new, com_src * (24.0 / box), tol=5e-2)
    c("resize:centre_of_mass_preserved", ok, f"{com_src} -> {com_new} (max rel err {err:.2e})")

    print("\n  reproducibility and the header sampling-rate fallback")
    header, _ = read_vol(out("header_sr"))
    c("header_sr:matches_explicit_sr", np.array_equal(header, resized), "--sr omitted == --sr given")
    workers, _ = read_vol(out("workers"))
    c("workers:matches_default", np.array_equal(workers, resized), "--num_workers does not change the result")

    print("\n  output naming")
    pair = os.path.join(runs, "pair")
    names = sorted(f for f in os.listdir(pair) if f.endswith(".mrc"))
    c("pair:both_volumes_written", len(names) == 2, ", ".join(names))
    c("pair:same_geometry",
      read_vol(os.path.join(pair, names[0]))[0].shape == read_vol(os.path.join(pair, names[1]))[0].shape)

    # The GUI binds a preprocess_volumes run to a downstream --vol only when a single
    # `*_preprocessed.mrc` matches; the duplicate-stem suffix must not escape that glob,
    # or a two-volume run would masquerade as an unambiguous one.
    dup = os.path.join(runs, "dup")
    import glob as _glob
    hits = sorted(os.path.basename(p) for p in _glob.glob(os.path.join(dup, "*_preprocessed.mrc")))
    c("dup:disambiguated_names", len(set(hits)) == 2, ", ".join(hits))
    c("dup:both_match_the_glob", len(hits) == 2,
      "a duplicate-stem run must not look single-volume to the GUI connector")

    return c.close()


# --------------------------------------------------------------------------- #
#  guards (the CLI must reject these, with a helpful message)                    #
# --------------------------------------------------------------------------- #
def _expect_failure(c, name, argv, fragment):
    proc = subprocess.run([sys.executable, "-c", _CLI_BOOTSTRAP] + argv,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=600)
    out = proc.stdout or ""
    c(f"{name}:rejected", proc.returncode != 0, f"exit {proc.returncode}")
    c(f"{name}:explains_why", fragment.lower() in out.lower(), f"expected {fragment!r}")


def check_particles_guards(args):
    c = Checks()
    md, work = args.source_md, args.runs
    base = ["preprocess_particles", "--md", md, "--sr", str(args.sr)]

    print("\n  invalid geometry and impossible requests are refused")
    _expect_failure(c, "odd_new_box_size", base + ["--new_box_size", "25", "--output_path", f"{work}/g1"],
                    "must be even")
    _expect_failure(c, "odd_crop_box_size", base + ["--crop_box_size", "31", "--output_path", f"{work}/g2"],
                    "must be even")
    _expect_failure(c, "no_op", base + ["--output_path", f"{work}/g3"], "nothing to do")
    _expect_failure(c, "zero_batch_size", base + ["--new_box_size", "24", "--batch_size", "0",
                                                  "--output_path", f"{work}/g4"], ">= 1")
    _expect_failure(c, "zero_read_workers", base + ["--new_box_size", "24", "--num_read_workers", "0",
                                                    "--output_path", f"{work}/g5"], ">= 1")
    _expect_failure(c, "ctf_without_ctf_metadata",
                    ["preprocess_particles", "--md", args.no_ctf_md, "--sr", str(args.sr),
                     "--new_box_size", "24", "--ctf_correction", "wiener", "--output_path", f"{work}/g6"],
                    "no CTF information")
    return c.close()


def check_volumes_guards(args):
    c = Checks()
    vol, work = args.source_vol, args.runs

    print("\n  invalid geometry and impossible requests are refused")
    _expect_failure(c, "no_op", ["preprocess_volumes", "--vol", vol, "--output_path", f"{work}/g1"],
                    "nothing to do")
    _expect_failure(c, "non_cubic_resize",
                    ["preprocess_volumes", "--vol", args.non_cubic, "--new_box_size", "16",
                     "--output_path", f"{work}/g2"], "not cubic")
    # A cubic, header-flagged particle stack: neither the ndim nor the cubic check can
    # catch it, so this pins the MRC-header guard specifically.
    _expect_failure(c, "particle_stack_as_volume",
                    ["preprocess_volumes", "--vol", args.stack, "--new_box_size", "16",
                     "--output_path", f"{work}/g3"], "is an image stack")
    return c.close()


CHECKS = {
    "particles": check_particles,
    "particles_guards": check_particles_guards,
    "volumes": check_volumes,
    "volumes_guards": check_volumes_guards,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", required=True, choices=sorted(CHECKS))
    parser.add_argument("--runs", required=True, help="Directory holding the CLI run outputs.")
    parser.add_argument("--box", type=int, default=48, help="Box size of the source data.")
    parser.add_argument("--sr", type=float, default=2.0, help="Sampling rate of the source data.")
    parser.add_argument("--source-md", default=None)
    parser.add_argument("--no-ctf-md", default=None)
    parser.add_argument("--source-vol", default=None)
    parser.add_argument("--non-cubic", default=None)
    parser.add_argument("--stack", default=None)
    args = parser.parse_args()
    sys.exit(CHECKS[args.check](args))


if __name__ == "__main__":
    main()
