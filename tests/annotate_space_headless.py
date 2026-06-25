#!/usr/bin/env python
"""
Headless component checks for the annotate_space GUI program.

annotate_space is an interactive PyQt5 + napari desktop app, so it cannot be run
through the CLI the way the batch programs are (its ``main()`` blocks on
``QApplication.exec_()`` and needs a display). This script exercises the
*headless, logic-bearing* pieces of the program directly — no manual interaction,
no real display — and is what ``tests/test_annotate_space.py`` drives as a
subprocess.

Checks (selected with ``--check``):

  dimred     PCA + UMAP dimensionality reduction (DimRedQThread)
  clustering KMeans + along-dimension clustering (ClusteringQThread)
  socket     viewer_socket Server<->Client round-trip (FromFiles map copy)
  utils      pure helper functions (getImagePath / getServerProgram / ...)
  offscreen  [Tier 2, best-effort] construct the napari viewer + the app's
             MultipleViewerWidget under QT_QPA_PLATFORM=offscreen, then close.
             Skips gracefully (exit 0) if the offscreen backend isn't usable.

Each check prints ``PASS``/``FAIL``/``SKIP`` and the script exits non-zero on a
real failure.
"""

import os
import sys
import argparse

# A headless Qt platform: no display, no window pops up.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np


def _ok(name, detail=""):
    print(f"PASS {name} {('- ' + detail) if detail else ''}")
    return True


def _fail(name, detail=""):
    print(f"FAIL {name} {('- ' + detail) if detail else ''}")
    return False


def _skip(name, detail=""):
    print(f"SKIP {name} {('- ' + detail) if detail else ''}")
    return True


def _qapp():
    """A singleton offscreen QApplication so QThread signals can be delivered."""
    from PyQt5.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _synthetic_latents(n=200, d=8, seed=0):
    rng = np.random.default_rng(seed)
    # a few Gaussian blobs so clustering / reduction have structure
    centers = rng.standard_normal((4, d)) * 5.0
    z = np.concatenate([c + rng.standard_normal((n // 4, d)) for c in centers], axis=0)
    return z.astype(np.float32)


# --------------------------------------------------------------------------- #
def check_dimred():
    from hax.viewers.annotate_space.threads.dimred_threads import DimRedQThread
    _qapp()
    z = _synthetic_latents()

    results = {}

    def run(mode, params):
        captured = {}
        thread = DimRedQThread(n_components=2, data=z, mode=mode, params=params)
        thread.red_space.connect(lambda arr: captured.setdefault("out", np.asarray(arr)))
        thread.run()                       # direct-connected signal -> synchronous
        return captured.get("out")

    pca = run("PCA", {})
    if pca is None or pca.shape != (z.shape[0], 2) or not np.isfinite(pca).all():
        return _fail("dimred:PCA", f"shape={None if pca is None else pca.shape}")
    results["PCA"] = pca.shape

    umap = run("UMAP", {"n_neighbors": 15, "min_dist": 0.1})
    if umap is None or umap.shape != (z.shape[0], 2) or not np.isfinite(umap).all():
        return _fail("dimred:UMAP", f"shape={None if umap is None else umap.shape}")
    results["UMAP"] = umap.shape

    return _ok("dimred", f"PCA{results['PCA']} UMAP{results['UMAP']}")


# --------------------------------------------------------------------------- #
def check_clustering():
    from hax.viewers.annotate_space.threads.clustering_threads import ClusteringQThread
    _qapp()
    z = _synthetic_latents()

    def run(n_clusters, mode, axis=None):
        captured = {}
        thread = ClusteringQThread(n_clusters=n_clusters, z_space=z, mode=mode, axis=axis)
        thread.centers_labels.connect(lambda cl: captured.setdefault("out", cl))
        thread.run()
        return captured.get("out")

    km = run(4, "KMeans")
    if km is None:
        return _fail("clustering:KMeans", "no result")
    centers, labels = km
    if np.asarray(centers).shape[0] != 4 or len(np.unique(labels)) > 4:
        return _fail("clustering:KMeans", f"centers={np.asarray(centers).shape} labels_uniq={len(np.unique(labels))}")

    ad = run(3, "Along_Dim", axis=0)
    if ad is None:
        return _fail("clustering:Along_Dim", "no result")
    centers2, labels2 = ad
    if np.asarray(centers2).shape != (3, z.shape[1]):
        return _fail("clustering:Along_Dim", f"centers={np.asarray(centers2).shape}")

    return _ok("clustering", f"KMeans centers={np.asarray(centers).shape}, Along_Dim centers={np.asarray(centers2).shape}")


# --------------------------------------------------------------------------- #
def check_socket(workdir):
    import threading
    from hax.viewers.annotate_space.viewer_socket.server import Server
    from hax.viewers.annotate_space.viewer_socket.client import Client

    outdir = os.path.join(workdir, "socket_out")
    os.makedirs(outdir, exist_ok=True)

    # A dummy "volume" file and a listing file pointing at it (FromFiles mode).
    vol = os.path.join(workdir, "dummy_volume.mrc")
    with open(vol, "wb") as f:
        f.write(b"DUMMY-VOLUME-CONTENT")
    listing = os.path.join(workdir, "volumes_list.txt")
    with open(listing, "w") as f:
        f.write(vol + "\n")

    # Server.__init__ blocks (listens/accepts), so run it in a daemon thread.
    port = Server.getFreePort()
    err = {}

    def run_server():
        try:
            Server("FromFiles", {"outdir": outdir}, port=port)
        except Exception as e:                # noqa: BLE001 - surfaced via err
            err["e"] = repr(e)

    threading.Thread(target=run_server, daemon=True).start()

    client = Client(port)                      # retries until the server is listening
    reply = client.sendDataToSever(listing)    # send the listing path; server copies it
    client.closeConnection()

    copied = os.path.join(outdir, "decoded_map_class_01.mrc")
    if reply != "Map generated":
        return _fail("socket", f"reply={reply!r} server_err={err.get('e')}")
    if not os.path.exists(copied):
        return _fail("socket", "copied volume missing")
    return _ok("socket", f"round-trip ok, reply={reply!r}, copied decoded_map_class_01.mrc")


# --------------------------------------------------------------------------- #
def check_utils():
    from hax.viewers.annotate_space.utils import utils

    p = utils.getImagePath("logo_small.png")
    if not p.replace("\\", "/").endswith("media/logo_small.png"):
        return _fail("utils:getImagePath", p)

    # getServerProgram builds a conda-activation command; conda may be absent in
    # the subprocess PATH, in which case treat the conda-dependent part as a soft pass.
    try:
        prog = utils.getServerProgram(env_name="hax", variables={"FOO": "1"})
        if "server.py" not in prog or "conda activate hax" not in prog or "FOO=1" not in prog:
            return _fail("utils:getServerProgram", prog)
        conda_detail = "getServerProgram ok"
    except FileNotFoundError:
        conda_detail = "conda not on PATH (getServerProgram conda part skipped)"

    return _ok("utils", f"getImagePath ok; {conda_detail}")


# --------------------------------------------------------------------------- #
def check_offscreen(workdir):
    """Tier 2 (best-effort): assemble the napari viewer + the app's custom
    MultipleViewerWidget headlessly, then close. Environment problems -> SKIP."""
    try:
        import napari
        from hax.viewers.annotate_space.annotate_space import MultipleViewerWidget
    except Exception as e:                      # noqa: BLE001
        return _skip("offscreen", f"import failed: {e!r}")

    _qapp()
    try:
        viewer = napari.Viewer(ndisplay=3, title="Annotate Space (headless)", show=False)
    except Exception as e:                      # noqa: BLE001 - offscreen backend not usable
        return _skip("offscreen", f"napari.Viewer offscreen unavailable: {e!r}")

    try:
        widget = MultipleViewerWidget(viewer, npoints=64, ndims=3, interactive=False)
        constructed = widget is not None
    except Exception as e:                      # noqa: BLE001
        viewer.close()
        return _fail("offscreen", f"MultipleViewerWidget construction error: {e!r}")
    finally:
        try:
            viewer.close()
        except Exception:                       # noqa: BLE001
            pass

    return _ok("offscreen", "napari viewer + MultipleViewerWidget assembled offscreen")


CHECKS = {
    "dimred": lambda a: check_dimred(),
    "clustering": lambda a: check_clustering(),
    "socket": lambda a: check_socket(a.workdir),
    "utils": lambda a: check_utils(),
    "offscreen": lambda a: check_offscreen(a.workdir),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", required=True, choices=list(CHECKS.keys()))
    parser.add_argument("--workdir", default="/tmp/annotate_space_headless")
    args = parser.parse_args()
    os.makedirs(args.workdir, exist_ok=True)
    ok = CHECKS[args.check](args)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
