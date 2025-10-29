#!/usr/bin/env python
# **************************************************************************
# *
# * Authors:     David Herreros Calero (dherreros@cnb.csic.es)
# *
# * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'scipion@cnb.csic.es'
# *
# **************************************************************************


import os
import shutil
from glob import glob, escape
import numpy as np
from scipy import signal
import pickle
import warnings
from packaging.version import parse as parse_version
from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.colors as colors
from xmipp_metadata.image_handler import ImageHandler

from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QSizePolicy, QApplication
from PyQt5.QtCore import QThread

from qtpy.QtWidgets import QSplitter, QTabWidget

import napari
from napari.components.viewer_model import ViewerModel
from napari._qt.layer_controls import QtLayerControlsContainer
from napari._qt.utils import _maybe_allow_interrupt
from napari.utils.notifications import show_warning, notification_manager
from napari.utils import progress
from napari.layers import Points

from magicgui.widgets import ComboBox, Container, Slider, Button

from hax.viewers.annotate_space.layers.layers import CustomPointsLayer, CustomImageLayer
from hax.viewers.annotate_space.threads.dimred_threads import DimRedQThread
from hax.viewers.annotate_space.threads.clustering_threads import ClusteringQThread
from hax.viewers.annotate_space.threads.pyqt_socket_threads import ServerQThread, ClientQThread
from hax.viewers.annotate_space.chimerax_connection.viewer_morph_chimerax import FlexMorphChimeraX
from hax.viewers.annotate_space.viewer_socket.server import Server
from hax.viewers.annotate_space.qt_widgets.tables import ParamTableWidget
from hax.viewers.annotate_space.qt_widgets.menus import install_canvas_context_menu, SavingMenuWidget, ClusteringMenuWidget
from hax.viewers.annotate_space.wrappers.wrappers import QtViewerWrap
from hax.viewers.annotate_space.utils.utils import getImagePath, getServerProgram, save_viewer_screenshot_with_dpi

NAPARI_GE_4_16 = parse_version(napari.__version__) > parse_version("0.4.16")


class MultipleViewerWidget(QSplitter):
    """The main widget of the example."""

    def __init__(self, viewer: napari.Viewer, npoints, ndims, interactive):
        super().__init__()
        self.viewer = viewer
        num_samples = 10000 if npoints > 10000 else npoints
        percentage = min(int(100 * num_samples / npoints), 100)

        if interactive:
            self.viewer_model1 = ViewerModel(title="map_view", ndisplay=3)

            self.qt_viewer1 = QtViewerWrap(self.viewer_model1, self.viewer_model1)
            self.qt_viewer1.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Maximum)

            self.dimred_methods_parameters_table = ParamTableWidget()
            self.dimred_update_parameters_button = Button(label="Update landscape with new parameters")
            self.dimred_update_parameters_button.visible = False

            self.tab_widget = QTabWidget()
            self.saving_menu_widget = SavingMenuWidget()
            self.clustering_menu_widget = ClusteringMenuWidget(ndims)
            dims = [f"Dim {dim + 1}" for dim in range(ndims)]
            items = [("X axis", dims), ("Y axis", dims), ("Z axis", dims)]
            value = ["Dim 1", "Dim 2", "Dim 3"]
            self.right_widgets = [ComboBox(choices=c, label=l, value=val) for [l, c], val in zip(items, value)]
            self.right_widgets.append(Slider(value=20, min=1, max=50, label="Landscape-Vol sigma"))
            self.right_widgets.append(Button(label="Extract selection to layer"))
            self.right_widgets.append(ComboBox(choices=[], label="# layer"))
            self.right_widgets.append(Button(label="Add selection to # layer"))
            self.right_widgets.append(Slider(value=percentage, min=0, max=100, label="Landscape downsampling"))
            self.right_widgets.append(ComboBox(choices=["Show Original", "Show PCA", "Show UMAP"], label="Landscape mode"))
            self.select_axis_container = Container(widgets=self.right_widgets)
            self.select_axis_container.native.layout().addWidget(self.dimred_methods_parameters_table)
            self.select_axis_container._dimred_methods_parameters_table = self.dimred_methods_parameters_table
            self.select_axis_container.append(self.dimred_update_parameters_button)
            w1 = QtLayerControlsContainer(self.viewer_model1)
            self.tab_widget.addTab(w1, "Volume display controls")
            self.tab_widget.addTab(self.clustering_menu_widget, "Landscape clustering controls")
            self.tab_widget.addTab(self.saving_menu_widget, "Saving controls")
            self.tab_widget.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Maximum)

            self.viewer.window.add_dock_widget(self.tab_widget, area="bottom")
            self.viewer.window.add_dock_widget(self.qt_viewer1, area="bottom")
            self.viewer.window.add_dock_widget(self.select_axis_container, area="right", name="Landscape controls")


class Annotate3D(object):

    def __init__(self, data, z_space, mode=None, path=".", interactive=True, **kwargs):
        # Prepare attributes
        self.class_inputs = kwargs
        self.data = data
        self.z_space = z_space
        self.path = path
        self.mode = mode
        self.prev_layers = None  # To keep track of old layers for callbacks
        self.last_selected = set()
        self.current_axis = [0, 1, 2]
        self.boxsize = 128
        env_name = kwargs.get("env_name", None)

        # Keyboard attributes
        self.control_pressed = False
        self.alt_pressed = False

        # Attributes to control callback flow
        self.allow_removing_cluster_layer = True
        self.allow_modifying_kmeans_layer = True

        # Create viewer
        self.view = napari.Viewer(ndisplay=3, title="Annotate Space")
        self.view.window._qt_window.setWindowIcon(QIcon(getImagePath(("logo_small.png"))))
        self.dock_widget = MultipleViewerWidget(self.view, self.data.shape[0], self.data.shape[1],
                                                interactive=interactive)

        # Load in view or interactive mode
        self.view.window._qt_viewer.dockLayerControls.setVisible(interactive)

        # PCA transformer (for PCA axis sampling)
        self.pca_transformer = PCA(n_components=self.data.shape[1])
        self.pca_transformer.fit(self.z_space)
        self.pca_data = self.pca_transformer.transform(self.z_space)

        # Scale data to box of side 300
        self.data = (self.boxsize - 1) * (self.data - np.amin(self.data)) / (np.amax(self.data) - np.amin(self.data))
        self.original_data = np.copy(self.data)  # For keeping it safe when changing dimred method

        # Downsample PC
        self.doing_dowsampling = False
        num_samples = 10000 if self.data.shape[0] > 10000 else self.data.shape[0]
        data, self.data_indices = downsample_point_cloud(self.data, num_samples)

        # Update landscape flag
        self.updating_landscape = False

        # Create KDTree
        self.kdtree_data = KDTree(self.data[:, :3])
        self.kdtree_z_pace = KDTree(self.z_space)

        # Set data in viewers
        points_layer = CustomPointsLayer(np.copy(data[:, :3]), size=1, shading='spherical',
                                                          border_width=0,
                                                          antialiasing=0,
                                                          blending="additive", name="Landscape")
        points_layer.editable = True
        self.dock_widget.viewer.add_layer(points_layer)

        # Set extra data layer (like priors) in viewer
        if "z_space_vol" in self.class_inputs:
            extra_data = np.loadtxt(self.class_inputs["z_space_vol"])
            _, inds = self.kdtree_z_pace.query(extra_data, k=1)
            inds = np.array(inds).flatten()
            extra_data = np.copy(self.data)[inds, :3]
            extra_layer = self.dock_widget.viewer.add_points(extra_data, size=1, shading='spherical',
                                                             border_width=0,
                                                             antialiasing=0,
                                                             visible=False,
                                                             blending="additive", name="Priors")
            extra_layer.editable = True

        # Create volume from data points
        indeces = np.round(self.data).astype(int)
        vol = np.zeros((self.boxsize, self.boxsize, self.boxsize))
        vol[indeces[:, 0], indeces[:, 1], indeces[:, 2]] += 1

        # Filter volume
        std = np.pi * np.sqrt(self.boxsize) / 10.0
        gauss_1d = signal.windows.gaussian(self.boxsize, std)
        kernel = np.einsum('i,j,k->ijk', gauss_1d, gauss_1d, gauss_1d)
        kernel = np.pad(kernel, (5, 5))
        vol = np.pad(vol, (5, 5))
        ft_vol = np.fft.fftshift(np.fft.fftn(vol))
        ft_vol_real = np.real(ft_vol) * kernel
        ft_vol_imag = np.imag(ft_vol) * kernel
        ft_vol = ft_vol_real + 1j * ft_vol_imag
        vol = np.real(np.fft.ifftn(np.fft.ifftshift(ft_vol)))[5:133, 5:133, 5:133]

        # Add volume and labels
        landscape_vol_layer = CustomImageLayer(vol, rgb=False, colormap="inferno", name="Landscape-Vol", opacity=0.5,
                                               blending='translucent_no_depth')
        self.view.add_layer(landscape_vol_layer)

        if interactive:
            if "boxsize" in self.class_inputs.keys():
                boxsize = int(self.class_inputs["boxsize"])
            else:
                boxsize = 64
            dummy_vol = np.zeros((boxsize, boxsize, boxsize))
            self.dock_widget.viewer_model1.add_image(dummy_vol, name="Map", rendering="mip")

        # Selections layers
        if interactive:
            self.reloadView()

        # Canvas context menu
        self.context_menu_canvas = install_canvas_context_menu(self.dock_widget.viewer)
        self.context_menu_canvas.set_action_callback("selection_to_chimerax", self.openSelectionWithChimeraX)

        # Add callbacks
        if interactive:
            self.dock_widget.viewer.mouse_drag_callbacks.append(self.lassoSelector)
            self.dock_widget.viewer.bind_key("Control", self.control_detection)
            self.dock_widget.viewer.bind_key("Alt", self.alt_detection)
            self.dock_widget.saving_menu_widget.save_btn.clicked.connect(self.saveSelections)
            self.dock_widget.saving_menu_widget.save_screenshot_button.clicked.connect(self.saveLandscapeScreenshot)
            self.dock_widget.clustering_menu_widget.cluster_btn.clicked.connect(self._compute_kmeans_fired)
            self.dock_widget.clustering_menu_widget.dimension_btn.clicked.connect(self._compute_dim_cluster_fired)
            self.dock_widget.clustering_menu_widget.morph_button.clicked.connect(self._morph_chimerax_fired)
            self.dock_widget.viewer.layers.events.inserted.connect(self.on_insert_add_callback)
            self.dock_widget.viewer.layers.events.removed.connect(self.on_removing_layer)
            self.dock_widget.right_widgets[0].changed.connect(lambda event: self.selectAxis(0, event))
            self.dock_widget.right_widgets[1].changed.connect(lambda event: self.selectAxis(1, event))
            self.dock_widget.right_widgets[2].changed.connect(lambda event: self.selectAxis(2, event))
            self.dock_widget.right_widgets[3].changed.connect(self.updateVolSigma)
            self.dock_widget.right_widgets[7].changed.connect(self.updateDownsampling)
            self.dock_widget.right_widgets[5].choices = self.getLayerChoices
            self.dock_widget.right_widgets[4].changed.connect(self.extractSelectionToLayer)
            self.dock_widget.right_widgets[6].changed.connect(self.addSelectionToLayer)
            self.dock_widget.right_widgets[8].changed.connect(self.dock_widget.dimred_methods_parameters_table._on_method_changed)
            self.dock_widget.right_widgets[8].changed.connect(self.changeVisiblityDimRedParamsButton)
            self.dock_widget.right_widgets[8].changed.connect(self.changeLandscapeView)
            self.dock_widget.dimred_update_parameters_button.clicked.connect(self.changeLandscapeView)

            # Worker threads
            self.thread_chimerax = None

            # Volume generation socket
            program = getServerProgram(env_name=env_name, variables={"CHIMERA_HOME": self.class_inputs["chimerax_binary"], "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"]})
            metadata = self.class_inputs
            metadata["outdir"] = self.path
            if self.mode == "FromFiles":
                with open(self.class_inputs["volumesPaths"], 'r') as f:
                    self.volumesPaths = f.read().splitlines()

            if metadata is not None:
                metadata_file = os.path.join(self.path, "metadata.p")
                with open(metadata_file, 'wb') as fp:
                    pickle.dump(metadata, fp, protocol=pickle.HIGHEST_PROTOCOL)

                # Start server
                self.port = Server.getFreePort()
                self.server = ServerQThread(program, metadata_file, self.mode, self.port, None)
                self.server.start()

                # Start client
                self.client = ClientQThread(self.port, self.path, self.mode)
                self.client.volume.connect(self.updateEmittedMap)
                self.client.chimera.connect(self.launchChimeraX)

            # Clustering threads initialization
            self.clustering_thread = None

            # Dimensionality reduction thread
            self.dimred_thread = None

        # Run viewer
        self.app = QApplication.instance()

        if interactive:
            self.app.aboutToQuit.connect(self.on_close_callback)

        with notification_manager, _maybe_allow_interrupt(self.app):
            self.app.exec_()

    # ---------------------------------------------------------------------------
    # Read functions
    # ---------------------------------------------------------------------------
    def reloadView(self):
        pathFile = os.path.join(self.path, "selections_layers")

        if os.path.isdir(pathFile):
            n_points = sum("Cluster" in directory.name for directory in os.scandir(pathFile))

            # Generate N evenly spaced values between 0 and 1
            values = np.linspace(0, 1, n_points)

            # Get the Viridis colormap
            viridis = plt.get_cmap('viridis', n_points)

            # Convert each color to HTML color code
            idc = 0
            cluster_colors = [colors.rgb2hex(viridis(value)) for value in values]

            for directory in os.scandir(pathFile):
                file = glob(escape(os.path.join(pathFile, directory.name, directory.name.removeprefix("SHL_"))) + "*")[0]
                if "Cluster" in directory.name:
                    text, features = None, None
                    size = 1
                    face_color = cluster_colors[idc]
                    extra_args = {"text": text, "size": size, "face_color": face_color, "features": features,
                                  "shading": 'spherical', "border_width": 0, "antialiasing": 0, "visible": False}
                    idc += 1
                elif "KMeans" in directory.name:
                    # Features
                    features = {'id': np.arange(n_points) + 1,}

                    # Text labels
                    text = {'string': 'Cluster {id:d}', 'size': 10, 'color': 'white', 'translation': -3}

                    size = 2
                    face_color = "#5500ff"
                    extra_args = {"text": text, "size": size, "face_color": face_color, "properties": features, "visible": True}
                else:
                    extra_args = {"visible": False}

                if ".tif" in file:
                    self.dock_widget.viewer.open(file, layer_type="labels", visible=False)
                elif ".csv" in file:
                    layer_type = "shapes" if "SHL_" in file else "points"
                    self.dock_widget.viewer.open(file, layer_type=layer_type, **extra_args)

                    if "Cluster" in file or "KMeans" in file:
                        layer = self.dock_widget.viewer.layers[-1]
                        custom_layer = CustomPointsLayer(data=layer.data, name=layer.name, **extra_args)
                        self.dock_widget.viewer.layers.remove(layer)
                        self.dock_widget.viewer.add_layer(custom_layer)

                # Cluster centers
                if "KMeans" in file:
                    self.z_center = self.dock_widget.viewer.layers[-1].data

        # Add callbacks
        for layer in self.dock_widget.viewer.layers:
            if not isinstance(layer, napari.layers.Shapes):
                if "Landscape-Vol" not in layer.name and len(layer.data.shape) == 2:
                    layer.events.data.connect(self.updateConformation)
                    layer.events.highlight.connect(self.updateConformation)

                if "KMeans" in layer.name:
                    self.current_kmeans_data = np.copy(layer.data)
                    layer.events.data.connect(self.on_kmeans_layer_data_removal)

    def readMap(self, file):
        map = ImageHandler().read(file).getData()
        return map

    # ---------------------------------------------------------------------------
    # Write functions
    # ---------------------------------------------------------------------------
    def writeVectorFile(self, vector):
        pathFile = os.path.join(self.path, self.vector_file)
        with open(pathFile, 'w') as fid:
            fid.write(' '.join(map(str, vector.reshape(-1))) + "\n")

    def saveSelections(self):
        pathFile = os.path.join(self.path, "selections_layers")

        if os.path.isdir(pathFile):
            shutil.rmtree(pathFile)

        os.mkdir(pathFile)

        with progress(self.dock_widget.viewer.layers) as pbr:
            for layer in pbr:
                    if "Landscape" not in layer.name and "Priors" not in layer.name:
                        points = layer.data

                        # If showing UMAP or PCA, we bring back the points to the original space
                        if not "Original" in self.dock_widget.right_widgets[8].value:
                            if isinstance(layer, napari.layers.Shapes):
                                original_points = np.array(points)
                                for idx, original_sub_points in enumerate(original_points):
                                    _, inds = self.kdtree_data.query(original_sub_points, k=1)
                                    inds = np.array(inds).flatten()
                                    original_points[idx] = np.copy(self.original_data[inds])
                                original_points = original_points.tolist()
                            else:
                                original_points = np.copy(points)
                                _, inds = self.kdtree_data.query(original_points, k=1)
                                inds = np.array(inds).flatten()
                                original_points = np.copy(self.original_data[inds])[:, self.current_axis]
                        else:
                            original_points = points

                        if len(points) > 0:
                            # Create a folder just for this layer
                            prefix = "SHL_" if isinstance(layer, napari.layers.Shapes) else ""
                            layer_folder = os.path.join(pathFile, prefix + layer.name)
                            os.mkdir(layer_folder)

                            self.allow_removing_cluster_layer = False
                            self.allow_modifying_kmeans_layer = False
                            self.updating_landscape = True
                            layer.data = original_points
                            layer.save(os.path.join(layer_folder, layer.name))
                            layer.data = points
                            layer.refresh()
                            self.updating_landscape = False
                            self.allow_removing_cluster_layer = True
                            self.allow_modifying_kmeans_layer = True

                            # Save metadata and representatives (volumes) of the layer:
                            if not isinstance(layer, napari.layers.Shapes):
                                _, inds = self.kdtree_data.query(points, k=1)
                                inds = np.array(inds).flatten()
                                z = self.z_space[inds]
                                self.client.save_to_file = True
                                if "Cluster" in layer_folder:
                                    self.client.z = z.mean(axis=0, keepdims=True)
                                    self.client.file_names = [""]
                                    self.client.start()
                                    self.client.wait()

                                    # Move map file to folder
                                    shutil.move(self.client.vol_file_template.format(1),
                                                os.path.join(layer_folder, "representative.mrc"))

                                    # Save indices (for indexing in "metadata")
                                    np.savetxt(os.path.join(layer_folder, "particle_indices.txt"), inds.astype(int), fmt='%d')

                                elif "KMeans" not in layer_folder:
                                    if "user_selection" in layer.metadata:
                                        self.client.z = z.mean(axis=0, keepdims=True)
                                        self.client.file_names = [""]
                                        self.client.start()
                                        self.client.wait()

                                        # Move map file to folder
                                        shutil.move(self.client.vol_file_template.format(1),
                                                    os.path.join(layer_folder, "representative.mrc"))

                                        # Save indices (for indexing in "metadata")
                                        np.savetxt(os.path.join(layer_folder, "particle_indices.txt"), inds.astype(int), fmt='%d')

                                    else:
                                        idx = 1
                                        if z.ndim == 1:
                                            z = z[None, ...]
                                        for z_line in z:
                                            self.client.z = z_line[None, ...]
                                            self.client.file_names = [""]
                                            self.client.start()
                                            self.client.wait()

                                            # Move map file to folder
                                            shutil.move(self.client.vol_file_template.format(1),
                                                        os.path.join(layer_folder, f"representative_point_{idx:02d}.mrc"))

                                            # Save indices (for indexing in "metadata")
                                            _, inds = self.kdtree_z_pace.query(z_line[None, ...], k=min(self.z_space.shape[0], 5000))
                                            inds = np.array(inds).flatten()
                                            np.savetxt(os.path.join(layer_folder, f"particle_indices_point_{idx:02d}.txt"), inds.astype(int), fmt='%d')

                                            idx += 1
                                self.client.save_to_file = False

    def saveLandscapeScreenshot(self):
        dpi = self.dock_widget.saving_menu_widget.screenshot_dpi.text()
        if dpi:
            dpi = int(dpi)
            transparent = self.dock_widget.saving_menu_widget.transparent_background.isChecked()
            save_viewer_screenshot_with_dpi(self.dock_widget.viewer, dpi=dpi, transparent_bg=transparent)
        else:
            show_warning('DPI not provided. Use the text box next to "Save landscape screenshot" to customize the '
                         'image resolution (DPI value must go from 1 to infinity - in general, 300 or 600 is enough to '
                         'get a high enough resolution for a figure).')

    # ---------------------------------------------------------------------------
    # Callbacks
    # ---------------------------------------------------------------------------
    def on_close_callback(self):
        # If viewer is closed, remove socket
        self.server.stop()

    def lassoSelector(self, viewer, event):
        # viewer = self.view

        # Lasso is activate with mouse wheel button
        if event.button == 3:

            # Stop warnings for now (due to weird Napari behaviour)
            warnings.filterwarnings('ignore')

            # Initialize the lasso path list
            lasso_path = []

            # Layer to select from
            layer = viewer.layers.selection._current

            if isinstance(layer, napari.layers.Points) or isinstance(layer, CustomPointsLayer):
                ndims = layer._view_data.shape[1]

                # Lasso layer
                lasso_layer = viewer.add_shapes(name='lasso', shape_type='path', edge_width=0.01, face_color=[0] * 4,
                                                edge_color="blue")

                # Add first clicked point to lasso path
                cursor = np.asarray(list(viewer.cursor.position))
                # cursor[0] = viewer.dims.current_step[0]
                lasso_path.append(cursor)
                yield

                # Keep on adding points on dragging
                while event.type == 'mouse_move' and not event.type == 'mouse_release':
                    cursor = (np.asarray(list(viewer.cursor.position)) +  # We might need to make scale smaller!
                              np.random.normal(loc=0, scale=0.001, size=3))  # Custom scale based on smallest distance?
                    # cursor[0] = viewer.dims.current_step[0]
                    lasso_path.append(cursor)
                    if len(lasso_path) > 1:
                        try:
                            lasso_layer.data = [np.array(lasso_path)]
                        except AttributeError:  # If we are selecting an empty selection just skip it
                            pass
                    yield

                # Once mouse is released, project points to 2D view if needed
                if ndims == 3:
                    points = layer._view_data
                    # points_proj = np.asarray([layer.world_to_data(x) for x in points])
                    # lasso_path_proj = np.asarray([layer.world_to_data(x) for x in lasso_path])
                    # projection_direction = np.asarray(layer.world_to_data(viewer.camera.view_direction))
                    points_proj = np.asarray(points)
                    lasso_path_proj = np.asarray(lasso_path)
                    projection_direction = np.asarray(viewer.camera.view_direction)
                    rot = rotation_matrix_from_vectors(projection_direction, np.asarray([0, 0, 1]))
                    points_proj = project_points(points_proj, projection_direction).dot(rot.T)[..., :2]
                    lasso_path_proj = project_points(lasso_path_proj, projection_direction).dot(rot.T)[..., :2]
                else:
                    points = layer.data
                    lasso_path_proj = np.asarray(lasso_path)
                    if points.shape[1] == 3:
                        ids = int(lasso_path_proj[0, viewer.dims.not_displayed])
                        ids = np.argwhere(np.logical_not(np.isclose(points[:, viewer.dims.not_displayed],
                                                                    ids, 1e-2)))[..., 0]
                        points_proj = points[..., viewer.dims.displayed]
                        points_proj[ids] *= 10000

                    else:
                        points_proj = points[..., viewer.dims.displayed]
                    lasso_path_proj = lasso_path_proj[..., viewer.dims.displayed]

                # Once mouse is released, create Lasso path and select points
                path = Path(lasso_path_proj)
                inside = path.contains_points(points_proj)
                selected_data = set(np.nonzero(inside)[0])

                if self.control_pressed:
                    layer.selected_data = layer.selected_data | selected_data
                elif self.alt_pressed:
                    layer.selected_data = layer.selected_data - selected_data
                else:
                    layer.selected_data = set(np.nonzero(inside)[0])
                lasso_path.clear()

                # Remove Lasso layer
                viewer.layers.remove(lasso_layer)

                # Keep selected layer active
                viewer.layers.selection.active = layer

            # Bring back warnings
            warnings.filterwarnings('default')

    def control_detection(self, event):
        # On key press
        self.control_pressed = True
        yield

        # On key release
        self.control_pressed = False

    def alt_detection(self, event):
        # On key press
        self.alt_pressed = True
        yield

        # On key release
        self.alt_pressed = False

    def _compute_kmeans_fired(self):
        n_clusters = int(self.dock_widget.clustering_menu_widget.cluster_num.text())

        # Auxilary function to update viewer layers
        def update_cluster_layers(centers_labels):
            landscape = self.data[:, self.current_axis]
            self.kmeans_data = []
            self.z_center = []

            # Remove previous clusters and kmeans
            self.allow_removing_cluster_layer = False
            self.allow_modifying_kmeans_layer = False
            layer_names = [layer.name for layer in self.dock_widget.viewer.layers]
            for layer_name in layer_names:
                if "Cluster_" in layer_name or "KMeans" in layer_name:
                    self.dock_widget.viewer.layers.remove(layer_name)
            self.allow_removing_cluster_layer = True
            self.allow_modifying_kmeans_layer = True

            centers, labels = centers_labels[0], centers_labels[1]
            self.z_center = np.copy(centers)
            self.interp_val = labels
            _, inds = self.kdtree_z_pace.query(centers, k=1)
            inds = np.array(inds).flatten()
            selected_data = np.copy(landscape[inds])
            self.kmeans_data.append(np.copy(self.data[inds]))

            # Features
            features = {
                'id': np.arange(selected_data.shape[0]) + 1,
            }

            # Text labels
            text = {
                'string': 'Cluster {id:d}',
                'size': 10,
                'color': 'white',
                'translation': -3,
            }

            kmeans_layer = CustomPointsLayer(selected_data, size=2, name="KMeans", features=features, text=text, face_color="#5500ff")
            self.dock_widget.viewer.add_layer(kmeans_layer)
            kmeans_layer.events.data.connect(self.on_kmeans_layer_data_removal)
            self.current_kmeans_data = np.copy(selected_data)

            # Add points for each cluster independently with colors
            self.dock_widget.viewer.layers["Landscape"].visible = False
            cm = plt.get_cmap("viridis")
            color_ids = np.linspace(0.0, 1.0, n_clusters)
            for label, color_id in zip(np.unique(self.interp_val), color_ids):
                self.kmeans_data.append(np.copy(self.data[self.interp_val == label]))
                cluster_points = np.copy(landscape[self.interp_val == label])
                color = np.asarray(cm(color_id))
                cluster_layer = CustomPointsLayer(cluster_points, size=1, name=f"Cluster_{label + 1:05d}",
                                                  visible=False,
                                                  shading='spherical', border_width=0, antialiasing=0,
                                                  face_color=color)
                self.dock_widget.viewer.add_layer(cluster_layer)

        def on_thread_finished():
            self.clustering_thread = None
            self.dock_widget.clustering_menu_widget.cluster_btn.setEnabled(True)
            self.dock_widget.clustering_menu_widget.dimension_btn.setEnabled(True)

        # Compute KMeans on QThread and save automatic selection
        self.dock_widget.clustering_menu_widget.cluster_btn.setEnabled(False)
        self.dock_widget.clustering_menu_widget.dimension_btn.setEnabled(False)
        self.clustering_thread = ClusteringQThread(n_clusters, self.z_space, "KMeans")
        self.clustering_thread.centers_labels.connect(update_cluster_layers)
        self.clustering_thread.finished.connect(on_thread_finished)
        self.clustering_thread.finished.connect(self.clustering_thread.deleteLater)
        self.clustering_thread.start()

    def _compute_dim_cluster_fired(self):
        axis = int(self.dock_widget.clustering_menu_widget.dimension_sel.currentText().replace("Dim ", "")) - 1
        n_clusters = int(self.dock_widget.clustering_menu_widget.cluster_num.text())

        # Auxilary function to update viewer layers
        def update_cluster_layers(centers_labels):
            landscape = self.data[:, self.current_axis]
            self.kmeans_data = []
            self.z_center = []

            # Remove previous clusters and kmeans
            self.allow_removing_cluster_layer = False
            self.allow_modifying_kmeans_layer = False
            layer_names = [layer.name for layer in self.dock_widget.viewer.layers]
            for layer_name in layer_names:
                if "Cluster_" in layer_name or "KMeans" in layer_name:
                    self.dock_widget.viewer.layers.remove(layer_name)
            self.allow_modifying_kmeans_layer = True
            self.allow_removing_cluster_layer = True

            self.interp_val = centers_labels[1].astype(int)

            # Cluster always along PCA space
            z_tr_data = self.pca_transformer.inverse_transform(centers_labels[0])
            _, inds = self.kdtree_z_pace.query(z_tr_data, k=1)
            inds = np.array(inds).flatten()
            group_means = np.copy(landscape[inds])
            self.z_center = np.copy(z_tr_data)

            # Features
            features = {
                'id': np.arange(group_means.shape[0]) + 1,
            }

            text = {
                'string': 'Cluster {id:d}',
                'size': 10,
                'color': 'white',
                'translation': -3,
            }

            kmeans_layer = CustomPointsLayer(group_means, size=2, name="KMeans along PCA {:d}".format(axis + 1),
                                             features=features, text=text, face_color="#5500ff")
            self.dock_widget.viewer.add_layer(kmeans_layer)
            kmeans_layer.events.data.connect(self.on_kmeans_layer_data_removal)
            self.current_kmeans_data = np.copy(group_means)

            # Add points for each cluster independently with colors
            self.dock_widget.viewer.layers["Landscape"].visible = False
            cm = plt.get_cmap("viridis")
            color_ids = np.linspace(0.0, 1.0, n_clusters)
            for label, color_id in zip(range(n_clusters), color_ids):
                if label in self.interp_val:
                    self.kmeans_data.append(np.copy(self.data[self.interp_val == label]))
                    cluster_points = np.copy(landscape[self.interp_val == label])
                else:
                    cluster_points = None
                color = np.asarray(cm(color_id))
                cluster_layer = CustomPointsLayer(cluster_points, size=1, name=f"Cluster_{label + 1:05d}", visible=False,
                                                  shading='spherical', border_width=0, antialiasing=0,
                                                  face_color=color)
                self.dock_widget.viewer.add_layer(cluster_layer)

        def on_thread_finished():
            self.clustering_thread = None
            self.dock_widget.clustering_menu_widget.cluster_btn.setEnabled(True)
            self.dock_widget.clustering_menu_widget.dimension_btn.setEnabled(True)

        # Compute KMeans on QThread and save automatic selection
        self.dock_widget.clustering_menu_widget.cluster_btn.setEnabled(False)
        self.dock_widget.clustering_menu_widget.dimension_btn.setEnabled(False)
        self.clustering_thread = ClusteringQThread(n_clusters, self.pca_data, "Along_Dim", axis=axis)
        self.clustering_thread.centers_labels.connect(update_cluster_layers)
        self.clustering_thread.finished.connect(on_thread_finished)
        self.clustering_thread.finished.connect(self.clustering_thread.deleteLater)
        self.clustering_thread.start()

    def _morph_chimerax_fired(self):
        # Morph maps in chimerax based on different ordering methods
        if self.thread_chimerax is None:
            layer = self.dock_widget.viewer.layers.selection._current
            if "Landscape" not in layer.name and "Cluster_" not in layer.name:
                points = layer.data if isinstance(layer.data, np.ndarray) else np.array(layer.data)
                if points.ndim == 3:
                    points = points.reshape((points.shape[0] * points.shape[1], points.shape[2]))
                if points.shape[0] > 0:
                    if "along PCA" in layer.name or "KMeans" in layer.name:
                        z_space = self.z_center
                        sel_names = ["vol_%03d" % (idx + 1) for idx in range(z_space.shape[0])]
                    else:
                        _, inds = self.kdtree_data.query(points, k=1)
                        inds = np.array(inds).flatten()
                        sel_names = ["vol_%03d" % (idx + 1) for idx in range(points.shape[0])]
                        z_space = self.z_space[inds]
                    if z_space.ndim == 1:
                        z_space = z_space[None, ...]

                    if self.client.isRunning():
                        show_warning(
                            "Previous conformation has not being generated yet, current selection will not be generated")
                    else:
                        if not self.mode == "FromFiles":
                            self.client.z = z_space
                        else:
                            _, idz = self.kdtree_data.query(points, k=1)
                            idz = np.array(idz).flatten()
                            sel_names = ["vol_%03d" % (idx + 1) for idx in range(points.shape[0])]
                            self.client.z = [self.volumesPaths[i] for i in idz]
                            self.client.z_coords = self.z_space[idz]
                        self.client.file_names = sel_names
                        self.client.start()

    def on_insert_add_callback(self, event):
        layer = event.value
        if layer.name != "lasso" and not isinstance(layer, napari.layers.Shapes):
            if len(layer.data.shape) == 2:
                layer.events.data.connect(self.updateConformation)
                layer.events.highlight.connect(self.updateConformation)

    def on_removing_layer(self, event):
        if self.allow_removing_cluster_layer:
            self.allow_modifying_kmeans_layer = False
            self.updating_landscape = True
            removed_layer = event.value
            if "Cluster_" in removed_layer.name:
                cluster_id = int(removed_layer.name.split("_")[-1]) - 1

                # Update saving data
                if hasattr(self, "z_center"):
                    self.z_center = np.delete(self.z_center, cluster_id, axis=0)

                # Modify data in KMeans layer
                layer = [layer for layer in self.dock_widget.viewer.layers if "KMeans" in layer.name]
                if len(layer) > 0:
                    layer = layer[0]
                    layer.data = np.delete(layer.data, cluster_id, axis=0)
                    self.current_kmeans_data = np.copy(layer.data)

                # Rename layers
                cluster_id = cluster_id + 2
                layer_names = [layer.name for layer in self.dock_widget.viewer.layers]
                while f"Cluster_{cluster_id}" in layer_names:
                    layer = self.dock_widget.viewer.layers[f"Cluster_{cluster_id:05d}"]
                    layer._fixed_name = f"Cluster_{cluster_id - 1:05d}"
                    cluster_id += 1

            self.allow_modifying_kmeans_layer = True
            self.updating_landscape = False

    def on_kmeans_layer_data_removal(self, event):
        if self.allow_modifying_kmeans_layer:
            self.allow_removing_cluster_layer = False
            kmeans_layer = [layer for layer in self.dock_widget.viewer.layers if "KMeans" in layer.name][0]
            current_data = kmeans_layer.data

            # Find the deleted points by comparing previous data with current data
            previous_indices = {tuple(point): idx for idx, point in enumerate(self.current_kmeans_data)}
            current_indices = {tuple(point): idx for idx, point in enumerate(current_data)}

            deleted_points = [point for point in previous_indices if point not in current_indices]
            deleted_ids = [previous_indices[point] for point in deleted_points]
            deleted_ids.sort(reverse=True)

            if deleted_points:
                for deleted_id in deleted_ids:
                    self.dock_widget.viewer.layers.remove(f"Cluster_{deleted_id + 1:05d}")

                    # Update saving data
                    if hasattr(self, "z_center"):
                        self.z_center = np.delete(self.z_center, deleted_id, axis=0)

                # Update cluster layer names
                for deleted_id in deleted_ids:
                    cluster_layers = [layer for layer in self.dock_widget.viewer.layers if "Cluster_" in layer.name]
                    for layer in cluster_layers:
                        cluster_id = int(layer.name.split("_")[-1])
                        if cluster_id > deleted_id + 1:
                            layer._fixed_name = f"Cluster_{cluster_id - 1:05d}"

            # Update KMeans layer labels
            text = {
                'string': [f"Cluster {idx + 1}" for idx in range(current_data.shape[0])],
                'size': 10,
                'color': 'white',
                'translation': -3,
            }
            kmeans_layer.text = text
            kmeans_layer.refresh()

            # Update the previous data
            self.current_kmeans_data = current_data.copy()

            self.allow_removing_cluster_layer = True

    def selectAxis(self, pos, event):
        axis = int(event.replace("Dim ", "")) - 1

        # Change point layers
        for layer in self.dock_widget.viewer.layers:
            if not isinstance(layer, napari.layers.Shapes):
                if len(layer.data.shape) == 2:
                    current_data = layer.data

                    if current_data.shape[0] != self.data.shape[0]:
                        _, inds = self.kdtree_data.query(current_data, k=1)
                        inds = np.array(inds).flatten()
                        layer.data[:, pos] = self.data[inds, axis]
                    else:
                        layer.data[:, pos] = self.data[:, axis]

                    layer.refresh()
            # else:  # TODO: What happens here? How should it be implemented?

        # Change volume
        layer = self.dock_widget.viewer.layers["Landscape-Vol"]
        boxsize = layer.data.shape[0]
        indeces = np.round(self.data).astype(int)
        vol = np.zeros((boxsize, boxsize, boxsize))
        vol[indeces[:, 0], indeces[:, 1], indeces[:, 2]] += 1

        # Filter volume
        std = np.pi * np.sqrt(boxsize) / self.dock_widget.right_widgets[3].get_value()
        gauss_1d = signal.windows.gaussian(boxsize, std)
        kernel = np.einsum('i,j,k->ijk', gauss_1d, gauss_1d, gauss_1d)
        kernel = np.pad(kernel, (5, 5))
        vol = np.pad(vol, (5, 5))
        ft_vol = np.fft.fftshift(np.fft.fftn(vol))
        ft_vol_real = np.real(ft_vol) * kernel
        ft_vol_imag = np.imag(ft_vol) * kernel
        ft_vol = ft_vol_real + 1j * ft_vol_imag
        vol = np.real(np.fft.ifftn(np.fft.ifftshift(ft_vol)))[5:133, 5:133, 5:133]
        layer.data = vol
        layer.refresh()

        # Update current axis
        self.current_axis[pos] = axis
        self.kdtree_data = KDTree(self.data[:, self.current_axis])

    def updateVolSigma(self, sigma):
        # Change volume
        layer = self.dock_widget.viewer.layers["Landscape-Vol"]
        boxsize = layer.data.shape[0]
        indeces = np.round(self.data).astype(int)
        vol = np.zeros((boxsize, boxsize, boxsize))
        vol[indeces[:, 0], indeces[:, 1], indeces[:, 2]] += 1

        # Filter volume
        std = np.pi * np.sqrt(boxsize) / sigma
        gauss_1d = signal.windows.gaussian(boxsize, std)
        kernel = np.einsum('i,j,k->ijk', gauss_1d, gauss_1d, gauss_1d)
        kernel = np.pad(kernel, (5, 5))
        vol = np.pad(vol, (5, 5))
        ft_vol = np.fft.fftshift(np.fft.fftn(vol))
        ft_vol_real = np.real(ft_vol) * kernel
        ft_vol_imag = np.imag(ft_vol) * kernel
        ft_vol = ft_vol_real + 1j * ft_vol_imag
        vol = np.real(np.fft.ifftn(np.fft.ifftshift(ft_vol)))[5:133, 5:133, 5:133]
        layer.data = vol
        layer.refresh()

    def updateDownsampling(self, percentage):
        self.doing_dowsampling = True
        num_samples = int(0.01 * percentage * self.data.shape[0])
        data, self.data_indices = downsample_point_cloud(self.data, num_samples)
        self.dock_widget.viewer.layers["Landscape"].data = data
        self.dock_widget.viewer.layers["Landscape"].selected_data = set()
        self.dock_widget.viewer.layers["Landscape"].refresh()
        self.doing_dowsampling = False

    def extractSelectionToLayer(self):
        choices = list(self.dock_widget.right_widgets[5].choices)
        total = len(choices)
        layer = self.dock_widget.viewer.layers.selection._current
        layer_name = f"Selected points {total + 1}"
        selected = layer.data[np.asarray(list(layer.selected_data)).astype(int)]
        self.dock_widget.viewer.add_points(selected, size=1, name=layer_name, visible=True,
                                           shading='spherical', border_width=0, antialiasing=0, blending="additive",
                                           face_color="white", metadata={"user_selection": True,})

    def addSelectionToLayer(self):
        layer_name = self.dock_widget.right_widgets[5].current_choice
        layer = self.dock_widget.viewer.layers[layer_name]
        sel_layer = self.dock_widget.viewer.layers.selection._current
        selected = sel_layer.data[np.asarray(list(sel_layer.selected_data)).astype(int)]
        layer.data = np.concatenate([layer.data, selected], axis=0)
        layer.refresh()

    def changeLandscapeView(self, event):
        if isinstance(event, str):
            method = event.split(" ")[-1]
        else:
            method = self.dock_widget.right_widgets[8].value.split(" ")[-1]
        n_clusters = self.original_data.shape[-1]

        def update_landscape_layer(data):
            self.updating_landscape = True

            data = (self.boxsize - 1) * (data - np.amin(data)) / (np.amax(data) - np.amin(data))

            # Update KDTree data
            kdtree_data = KDTree(data[:, self.current_axis])

            # Update landscape
            landscape_data = data[self.data_indices][:, self.current_axis]
            self.dock_widget.viewer.layers["Landscape"].data = landscape_data
            self.dock_widget.viewer.layers["Landscape"].selected_data = set()
            self.dock_widget.viewer.layers["Landscape"].refresh()

            # Change volume
            layer = self.dock_widget.viewer.layers["Landscape-Vol"]
            boxsize = layer.data.shape[0]
            indeces = np.round(data).astype(int)
            vol = np.zeros((boxsize, boxsize, boxsize))
            vol[indeces[:, 0], indeces[:, 1], indeces[:, 2]] += 1

            # Filter volume
            std = np.pi * np.sqrt(boxsize) / self.dock_widget.right_widgets[3].value
            gauss_1d = signal.windows.gaussian(boxsize, std)
            kernel = np.einsum('i,j,k->ijk', gauss_1d, gauss_1d, gauss_1d)
            kernel = np.pad(kernel, (5, 5))
            vol = np.pad(vol, (5, 5))
            ft_vol = np.fft.fftshift(np.fft.fftn(vol))
            ft_vol_real = np.real(ft_vol) * kernel
            ft_vol_imag = np.imag(ft_vol) * kernel
            ft_vol = ft_vol_real + 1j * ft_vol_imag
            vol = np.real(np.fft.ifftn(np.fft.ifftshift(ft_vol)))[5:133, 5:133, 5:133]
            layer.data = vol
            layer.refresh()

            # Update each layer
            self.allow_removing_cluster_layer = False
            self.allow_modifying_kmeans_layer = False
            for layer in self.dock_widget.viewer.layers:
                if "Landscape" not in layer.name:
                    layer_points = layer.data if isinstance(layer.data, np.ndarray) else np.array(layer.data)
                    if layer_points.ndim == 2:
                        layer_points = layer_points[np.newaxis, :, :]
                    for idx, points in enumerate(layer_points):
                        _, inds = self.kdtree_data.query(points, k=1)
                        inds = np.array(inds).flatten()
                        layer_points[idx] = np.copy(data[inds])[:, self.current_axis]
                    if isinstance(layer, napari.layers.Shapes):
                        layer_points = layer_points.tolist()
                    elif isinstance(layer, napari.layers.Points) or isinstance(layer, CustomPointsLayer):
                        layer_points = np.squeeze(layer_points)
                    layer.data = layer_points
                    layer.refresh()
            self.allow_removing_cluster_layer = True
            self.allow_modifying_kmeans_layer = True

            # Update data and its KDTree
            self.data = data
            self.kdtree_data = kdtree_data

            self.updating_landscape = False

        def on_thread_finished():
            self.dimred_thread = None

        # Compute KMeans on QThread and save automatic selection
        if method == "Original":
            update_landscape_layer(self.original_data)
        else:
            self.dimred_thread = DimRedQThread(n_clusters, self.z_space, method, self.dock_widget.dimred_methods_parameters_table.current_kwargs(only_changed=True))
            self.dimred_thread.red_space.connect(update_landscape_layer)
            self.dimred_thread.finished.connect(on_thread_finished)
            self.dimred_thread.finished.connect(self.dimred_thread.deleteLater)
            self.dimred_thread.start()

    def changeVisiblityDimRedParamsButton(self, event):
        method = event.split(" ")[-1]
        self.dock_widget.dimred_update_parameters_button.visible = not "Original" in method

    def getLayerChoices(self, extra):
        choices = []
        for layer in self.dock_widget.viewer.layers:
            if "user_selection" in layer.metadata:
                if layer.metadata["user_selection"]:
                    choices.append(layer.name)
        return choices

    def openSelectionWithChimeraX(self, viewer):
        if self.thread_chimerax is None:
            selected_layer = viewer.layers.selection.active
            if selected_layer:
                selected_points = self.data[list(selected_layer.selected_data)]
                if selected_points.ndim == 1:
                    selected_points = selected_points[None, ...]
                selected_names = ["vol_%03d" % (idx + 1) for idx in range(selected_points.shape[0])]

                if selected_points.shape[0] > 20:
                    show_warning(f"Cannot open selection with ChimeraX. To avoud generating too many states, we allow a "
                                 f"maximum of 20 points in the selection (i.e. 20 states to be shown in ChimeraX at once)."
                                 f"The number of points currently selected was {selected_points.shape[0]}.")

                _, inds = self.kdtree_data.query(selected_points, k=1)
                inds = np.array(inds).flatten()
                selected_z_space = self.z_space[inds]

                if selected_z_space.ndim == 1:
                    selected_z_space = selected_z_space[None, ...]

                if not self.mode == "FromFiles":
                    self.client.z = selected_z_space
                else:
                    _, idz = self.kdtree_data.query(selected_points, k=1)
                    idz = np.array(idz).flatten()
                    selected_names = ["vol_%03d" % (idx + 1) for idx in range(selected_points.shape[0])]
                    self.client.z = [self.volumesPaths[i] for i in idz]
                    self.client.z_coords = self.z_space[idz]
                self.client.file_names = selected_names
                self.client.is_chimera_signal = True
                self.client.start()

    # ---------------------------------------------------------------------------
    # Update map functions
    # ---------------------------------------------------------------------------
    def updateConformation(self, event):
        if self.updating_landscape:
            return

        if event.type != "highlight" and not self.doing_dowsampling:
            # Update real time conformation
            pos = event.value
            layer_idx = event.index

            if event.type == "data" and (event.action.name == "ADDING" or event.action.name == "CHANGING"):
                return

            if self.prev_layers is not None and layer_idx < len(self.prev_layers):
                if self.prev_layers[layer_idx].shape[0] > pos.shape[0]:
                    return
                prev_layer = self.prev_layers[layer_idx]
                nrows, ncols = pos.shape
                dtype = {'names': ['f{}'.format(i) for i in range(ncols)],
                         'formats': ncols * [pos.dtype]}
                pos = np.setdiff1d(pos.view(dtype), prev_layer.view(dtype))
                pos = pos.view(prev_layer.dtype).reshape(-1, ncols)
        else:
            layer = event.source
            if layer.mode == 'select':
                selected = list(layer.selected_data)
                if selected != self.last_selected and len(selected) > 0:
                    self.last_selected = selected
                    pos = layer.data[selected]
                    if pos.ndim == 2 and pos.shape[0] > 1:
                        pos = np.mean(pos, axis=0)
                        pos = pos[None, ...]
                else:
                    return
            else:
                return

        _, ind = self.kdtree_data.query(pos, k=1)
        ind = np.array(ind).flatten()[0]

        if self.client.isRunning():
            show_warning("Previous conformation has not being generated yet, current selection will not be generated")
        else:
            if not self.mode == "FromFiles":
                self.client.z = self.z_space[ind, :][None, ...]
            else:
                idz = [np.flatnonzero((vec == self.z_space).all(1))[0] for vec in self.z_space[ind, :][None, ...]]
                self.client.z = [self.volumesPaths[i] for i in idz]
            self.client.file_names = [""]
            self.client.start()

    # ---------------------------------------------------------------------------
    # Worker generation
    # ---------------------------------------------------------------------------
    def createThreadChimeraX(self, *args):
        self.thread_chimerax = QThread()
        self.worker = FlexMorphChimeraX(*args)
        self.worker.moveToThread(self.thread_chimerax)
        self.thread_chimerax.started.connect(self.worker.showSalesMan)
        self.worker.finished.connect(self.thread_chimerax.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread_chimerax.finished.connect(self.thread_chimerax.deleteLater)
        self.thread_chimerax.finished.connect(self.removeThreadChimeraX)

    def removeThreadChimeraX(self):
        self.thread_chimerax = None

    def updateEmittedMap(self, map):
        layer = self.dock_widget.viewer_model1.layers[0]
        data_is_empty = np.sum(layer.data[:]) == 0.0
        layer.data = map
        self.prev_layers = [layer.data.copy() for layer in self.dock_widget.viewer.layers]

        if data_is_empty:
            self.dock_widget.viewer_model1.reset_view()
            layer.iso_threshold = 3.0 * np.std(map[map >= 0.0])

        layer.contrast_limits = [0.0, np.amax(map)]

    def launchChimeraX(self):
        sel_names = self.client.file_names
        if not self.mode == "FromFiles":
            z = self.client.z
        else:
            z = self.client.z_coords
        self.createThreadChimeraX(z, sel_names, self.path, self.class_inputs["chimerax_binary"])
        self.thread_chimerax.start()

# ---------------------------------------------------------------------------
# Utils functions
# ---------------------------------------------------------------------------
def project_points(points, normal):
    """
    Projects the points with coordinates x, y, z onto the plane
    defined by a*x + b*y + c*z = 1
    """
    a, b, c = normal[0], normal[1], normal[2]
    vector_norm = a * a + b * b + c * c
    normal_vector = np.array([a, b, c]) / np.sqrt(vector_norm)
    point_in_plane = np.array([a, b, c]) / vector_norm

    points_from_point_in_plane = points - point_in_plane
    proj_onto_normal_vector = np.dot(points_from_point_in_plane,
                                     normal_vector)
    proj_onto_plane = (points_from_point_in_plane -
                       proj_onto_normal_vector[:, None] * normal_vector)

    return point_in_plane + proj_onto_plane

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def downsample_point_cloud(points, num_samples):
    indices = np.random.choice(np.arange(points.shape[0], dtype=int), num_samples, replace=False)
    return points[indices], indices


def main():
    import argparse
    import pathlib
    from hax.utils import bcolors

    # Input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--z_space', type=str, required=True,
                        help=f"The latent space estimated by any method and saved in {bcolors.ITALIC}txt{bcolors.ENDC} or "
                             f"{bcolors.ITALIC}npy{bcolors.ENDC} format.")
    parser.add_argument('--z_space_reduced', type=str, required=False,
                        help=f"A reduced dimensionality version of your input latent space. If not provided, the viewer will use "
                             f"the original latent space provided through {bcolors.ITALIC}z_space{bcolors.ENDC} will be displayed. "
                             f"Valid extensions include {bcolors.ITALIC}.txt{bcolors.ENDC} and {bcolors.ITALIC}.npy{bcolors.ENDC}")
    parser.add_argument('--path', type=str, required=True,
                        help="Path were viewer outputs will be saved.")
    parser.add_argument('--only_view', action='store_true',
                        help=f"When provided, the viewer will only display the conformational landscape. State generation capabilities are "
                             f"stopped. When neither {bcolors.ITALIC}env_name{bcolors.ENDC} nor {bcolors.ITALIC}volumes_path{bcolors.ENDC} are "
                             f"provided, the viewer will always use this mode.")
    parser.add_argument('--chimerax_binary', type=str, required=False,
                        help="If you want the viewer to connect with ChimeraX, please, provide here the absolute path to the ChimeraX binary from "
                             "your installation.")
    parser.add_argument('--env_name', type=str, required=False, default="hax",
                        help="When generating conformational states with a given software, you will need to provide here the name of the Conda environment "
                             "where that software is installed")
    parser.add_argument('--server_functions_path', type=str, required=False,
                        help=f"When parameter {bcolors.ITALIC}env_name{bcolors.ENDC} is provided, you must provide as well this parameter. Here, you must specify "
                             f"a path to a Python {bcolors.ITALIC}.py{bcolors.ENDC} file containing a class with two methods: {bcolors.ITALIC}prepare_heterogeneity_program{bcolors.ENDC}"
                             f" and {bcolors.ITALIC}decode_state_from_latent{bcolors.ENDC}. These funcitons allows to setup and execute the heterogeneity method used to "
                             f"estimate the landscape provided through {bcolors.ITALIC}z_space{bcolors.ENDC}, so that the viewer can generate conformations directly "
                             f"from this method. An example on how this class must be defined ca be found in this package source code available at: "
                             f"{bcolors.ITALIC}hax/viewers/server_loading_functions/load_model.py{bcolors.ENDC}.")
    parser.add_argument('--volumes_path', type=str, required=False,
                        help=f"If {bcolors.ITALIC}env_name{bcolors.ENDC} is not provided, you can still visualize a set of pre-saved conformation using this parameter. Here "
                             f"you will pass a path to a {bcolors.ITALIC}.txt{bcolors.ENDC} file containing as many volume paths as points has the landscape provided in "
                             f"{bcolors.ITALIC}z_space{bcolors.ENDC}. {bcolors.WARNING}NOTE that the points in your landscape and the volume paths stored in the "
                             f"txt file MUST have the same order.{bcolors.ENDC}")
    parser.add_argument_group(description=f"Apart from the previous arguments, the viewer is prepare to receive any number of additional custom input parameters. You can "
                                          f"pass them following the syntax: {bcolors.ITALIC}--my_parameter VALUE{bcolors.ENDC} where {bcolors.ITALIC}VALUE{bcolors.ENDC} "
                                          f"can be either a number or a string. Passing custom parameters is useful when parameter {bcolors.ITALIC}server_functions_path{bcolors.ENDC}"
                                          f" is provided, as it will allow you to tell the viewer which are the parameters needed to run the functions stored in the Python script "
                                          f"input through {bcolors.ITALIC}server_functions_path{bcolors.ENDC}.")

    # We can pass any additional argument to the viewer so that it can be easily adapted to any software.
    # In these cases, the parameter "server_functions_path" is MANDATORY, as it contains the path to a class with two methods
    # "prepare_heterogeneity_program" and "decode_state_from_latent" needed by the viewer to load/generate states in real
    # time for any method. The path MUST point always to a .py file containing the previous class.

    def float_or_str(s):
        try:
            return float(s)
        except ValueError:
            return s

    parsed, unknown = parser.parse_known_args()
    for arg in unknown:
        if arg.startswith(("-", "--")):
            # you can pass any arguments to add_argument
            parser.add_argument(arg.split('=')[0], type=float_or_str)

    args = parser.parse_args()

    # Read and generate data
    if pathlib.Path(args.z_space).suffix == ".txt":
        load_fn = np.loadtxt
    elif pathlib.Path(args.z_space).suffix == ".npy":
        load_fn = np.load
    else:
        raise ValueError(f"{bcolors.ITALIC}{args.z_space}{bcolors.ENDC} extension must be either "
                         f"{bcolors.ITALIC}.txt{bcolors.ENDC} or {bcolors.ITALIC}npy{bcolors.ENDC}")
    z_space = load_fn(args.z_space)

    if args.z_space_reduced is not None:
        if pathlib.Path(args.z_space).suffix == ".txt":
            load_fn = np.loadtxt
        elif pathlib.Path(args.z_space).suffix == ".npy":
            load_fn = np.load
        else:
            raise ValueError(f"{bcolors.ITALIC}{args.z_space_reduced}{bcolors.ENDC} extension must be either "
                             f"{bcolors.ITALIC}.txt{bcolors.ENDC} or {bcolors.ITALIC}npy{bcolors.ENDC}")
        data = load_fn(args.z_space_reduced)
    else:
        data = z_space

    # Input
    input_dict = vars(args)
    input_dict['data'] = data
    input_dict['z_space'] = z_space
    input_dict['interactive'] = not args.only_view

    if args.volumes_path is not None:
        input_dict['volumesPath'] = args.volumes_path

    if args.volumes_path is None and args.env_name is None:
        input_dict["interactive"] = False

    # Initialize volume slicer
    Annotate3D(**input_dict)

if __name__ == "__main__":
    main()
