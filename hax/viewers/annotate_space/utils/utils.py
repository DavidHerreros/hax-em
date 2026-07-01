

import os
import subprocess
import numpy as np
from PIL import Image

from PyQt5.QtWidgets import QFileDialog

import hax.viewers.annotate_space.viewer_socket.server as server


def getImagePath(image_name):
    """Returns the absolute path to an image within the package."""
    module_dir = os.path.dirname(__file__)  # Get the directory of this module
    image_path = os.path.join(module_dir, '..', 'media', image_name)
    return image_path

def getServerProgram(env_name=None, variables=None):
    """Build the command to call the server script."""
    program = "python " + server.__file__

    if variables is None:
        variables = ""
    else:
        variables = ' '.join(f"{key}={value}" for key, value in variables.items())

    program = f"{getCondaActivationCommand()} && conda activate {env_name} && {variables} {program}"

    return program

def getCondaBase():
    try:
        conda_base = subprocess.check_output("conda info --base", shell=True, text=True).strip()
        return conda_base
    except subprocess.CalledProcessError as e:
        print(f"Error finding Conda base: {e}")
        return None

def getCondaActivationCommand():
    return f'eval "$({getCondaBase()}/bin/conda shell.bash hook)"'

def _logical_canvas_wh(viewer):
    """Return (width, height) of the napari canvas across napari versions."""
    s = viewer.window._qt_viewer.canvas.size
    if callable(s):                                 # old napari: size() -> QSize
        s = s()
    if hasattr(s, "width") and callable(s.width):   # QSize-like object
        return int(s.width()) or 1, int(s.height()) or 1
    return int(s[0]) or 1, int(s[1]) or 1           # tuple/list (newer napari)

def save_viewer_screenshot_with_dpi(
    viewer,
    dpi=300,
    *,
    width_in=None,
    height_in=None,
    base_dpi=100,
    scale=None,
    canvas_only=True,
    flash=False,
    transparent_bg=False,
    bg_tolerance=50,
):
    """
    napari screenshot saved with a custom DPI, mimicking matplotlib savefig:
    more pixels, same framing, same aspect ratio.
    """
    # Native screenshot first: its array shape gives the unambiguous (H, W).
    base = viewer.screenshot(canvas_only=canvas_only, flash=False)
    nat_h, nat_w = base.shape[:2]
    aspect = nat_h / nat_w  # height / width, from the real image

    if width_in is not None or height_in is not None:
        if width_in is not None and height_in is None:
            target_w = int(round(width_in * dpi))
            target_h = int(round(target_w * aspect))
        elif height_in is not None and width_in is None:
            target_h = int(round(height_in * dpi))
            target_w = int(round(target_h / aspect))
        else:
            target_w = int(round(width_in * dpi))
            target_h = int(round(height_in * dpi))
    else:
        s = scale if scale is not None else (dpi / base_dpi)
        target_w = int(round(nat_w * s))
        target_h = int(round(nat_h * s))

    target_w, target_h = max(1, target_w), max(1, target_h)

    # Scale camera zoom by the same linear factor so framing is preserved.
    zoom_factor = target_w / nat_w
    camera = viewer.camera
    old_zoom = camera.zoom
    try:
        camera.zoom = old_zoom * zoom_factor
        # napari `size` is (height, width)
        arr = viewer.screenshot(
            size=(target_h, target_w),
            canvas_only=canvas_only,
            flash=flash,
        )
    finally:
        camera.zoom = old_zoom

    # --- optional transparency: knock out the canvas background color ---
    if transparent_bg:
        h, w = arr.shape[:2]
        corners = np.array([arr[0, 0], arr[0, -1], arr[-1, 0], arr[-1, -1]], dtype=np.int16)
        bg_col = np.median(corners, axis=0).astype(np.uint8)
        if arr.shape[2] == 3:
            rgba = np.concatenate([arr, 255 * np.ones((h, w, 1), dtype=np.uint8)], axis=2)
        else:
            rgba = arr.copy()
        diff = np.abs(rgba[:, :, :3].astype(np.int16) - bg_col[:3].astype(np.int16))
        mask = (diff <= bg_tolerance).all(axis=2)
        rgba[mask, 3] = 0
        arr = rgba

    # --- file dialog ---
    parent = getattr(viewer.window, "_qt_window", None)
    filters = "PNG (*.png);;JPEG (*.jpg *.jpeg);;TIFF (*.tif *.tiff)"
    path, selected_filter = QFileDialog.getSaveFileName(parent, "Save screenshot", "", filters)
    if not path:
        return None
    root, ext = os.path.splitext(path)
    ext = (ext or ".png").lower()
    path = root + ext
    if transparent_bg and ext in (".jpg", ".jpeg"):
        path = root + ".png"
        ext = ".png"

    # --- save with DPI (and alpha if present) ---
    im = Image.fromarray(arr)
    save_kwargs = {}
    if ext in (".jpg", ".jpeg"):
        save_kwargs["quality"] = 95
    im.save(path, dpi=(dpi, dpi), **save_kwargs)
    return path