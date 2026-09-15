import random
import sys
from contextlib import closing
from functools import partial

import numpy as np
from scipy.special import gammaln

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from sklearn.decomposition import PCA

import pynndescent
from cuml.neighbors.nearest_neighbors import NearestNeighbors

from hax.utils.loggers import bcolors


def _pbar(steps, **kwargs):
    from tqdm import tqdm
    return tqdm(steps, file=sys.stdout, ascii=" >=", colour="green",
                bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}", **kwargs)


class FreeEnergyMLP(nnx.Module):
    """Small MLP mapping a latent vector z to a scalar free energy G = f(z)"""

    def __init__(self, lat_dim, hidden=128, n_layers=3, *, rngs: nnx.Rngs):
        dims = [lat_dim] + [hidden] * n_layers
        self.hidden = nnx.List([nnx.Linear(d_in, d_out, rngs=rngs) for d_in, d_out in zip(dims[:-1], dims[1:])])
        self.out = nnx.Linear(hidden, 1, rngs=rngs)

    def __call__(self, z):
        for layer in self.hidden:
            z = nnx.silu(layer(z))
        return self.out(z)[..., 0]


class SupportWall(nnx.Module):
    """Force smooth and saturated output outside the data"""

    def __init__(self, in_dim, n_centres, width=0.1):
        n_centres = max(1, int(n_centres))
        self.centres = nnx.Variable(jnp.zeros((n_centres, in_dim), jnp.float32))
        self.radii = nnx.Variable(jnp.ones((n_centres,), jnp.float32))
        self.enabled = nnx.Variable(jnp.zeros((), jnp.float32))
        self.width = float(width)

    def fit(self, X, radius_percentile=98.0, radius_margin=1.25, seed=0):
        """Place the centres with k-means and give each the (margin-scaled) percentile radius of its members"""
        from sklearn.cluster import MiniBatchKMeans

        X = np.asarray(X, np.float32)
        M = self.centres[...].shape[0]
        n_clusters = min(M, len(X))
        km = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed, n_init=3, batch_size=4096).fit(X)
        labels = km.labels_
        dist = np.linalg.norm(X - km.cluster_centers_[labels], axis=1)

        order = np.argsort(labels, kind="stable")
        bounds = np.searchsorted(labels[order], np.arange(n_clusters + 1))
        radii = np.full(n_clusters, np.nan, np.float32)
        for c in range(n_clusters):
            members = dist[order[bounds[c]:bounds[c + 1]]]
            if len(members) > 1:
                radii[c] = np.percentile(members, radius_percentile)
        fallback = np.nanmedian(radii) if np.isfinite(radii).any() else 1.0
        radii = np.where(np.isfinite(radii) & (radii > 0), radii, fallback)

        # Pad (n_clusters < M when there are fewer points than centres) by repeating the last centre
        centres = np.concatenate([km.cluster_centers_, np.repeat(km.cluster_centers_[-1:], M - n_clusters, 0)])
        radii = np.concatenate([radii, np.repeat(radii[-1:], M - n_clusters)])
        self.centres[...] = jnp.asarray(centres, jnp.float32)
        self.radii[...] = jnp.asarray(radius_margin * radii, jnp.float32)
        self.enabled[...] = jnp.ones((), jnp.float32)

    def _scaled_distances(self, x):
        diff = x[:, None, :] - self.centres[...][None]
        return jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + 1e-12) / self.radii[...][None]

    def local_radius(self, x):
        """Radius of the closest centre (in input units) for each point"""
        return self.radii[...][jnp.argmin(self._scaled_distances(x), axis=1)]

    def __call__(self, x):
        d_min = jnp.min(self._scaled_distances(x), axis=1)
        return self.enabled[...] * jax.nn.sigmoid((d_min - 1.0) / self.width)


class EnergyHead(nnx.Module):
    """MLP energy saturated outside the data: G(x) = (1 - s) * mlp(x) + s * height, with s = wall(x)"""

    def __init__(self, in_dim, hidden=128, n_layers=3, n_wall_centres=256, *, rngs: nnx.Rngs):
        self.mlp = FreeEnergyMLP(in_dim, hidden=hidden, n_layers=n_layers, rngs=rngs)
        self.wall = SupportWall(in_dim, n_wall_centres)
        self.height = nnx.Variable(jnp.zeros((), jnp.float32))

    def __call__(self, x):
        s = self.wall(x)
        return (1.0 - s) * self.mlp(x) + s * self.height[...]


class _Opaque:
    """Holds objects that must stay out of the module state (hetsiren graphdef/state, searchers, jits)"""

    def __init__(self, value=None):
        self.value = value


class PathFinder(nnx.Module):
    # Slots of the ``fitted`` flag vector: which small networks have been trained
    _FIT_G, _FIT_E, _FIT_Z = 0, 1, 2

    def __init__(self, lat_dim, n_particles=None, feat_dim=None, max_points=2000, k=100,
                 hidden_energy=128, n_layers_energy=3, hidden_latent=256, n_layers_latent=3,
                 n_wall_centres=256, *, rngs: nnx.Rngs):

        self.config = {"_target_": "hax.utils.path_finder.PathFinder", "lat_dim": int(lat_dim),
                       "n_particles": n_particles, "feat_dim": feat_dim, "max_points": int(max_points),
                       "k": int(k), "hidden_energy": int(hidden_energy), "n_layers_energy": int(n_layers_energy),
                       "hidden_latent": int(hidden_latent), "n_layers_latent": int(n_layers_latent),
                       "n_wall_centres": int(n_wall_centres)}

        self.lat_dim, self.max_points, self.k = int(lat_dim), int(max_points), int(k)

        # Networks: G = f(z) (latent -> energy), E = f(F) (features -> energy), z = f(F) (features -> latent)
        self.G_model = self._new_energy_head(self.lat_dim, rngs)
        self.E_model = nnx.data(None)

        # Instantiate variables created in PathFinder to allow saving them
        self.idx = nnx.Variable(jnp.zeros((self.max_points,), jnp.int32))       # mobile Gaussian indices
        self.w = nnx.Variable(jnp.zeros((self.max_points,), jnp.float32))       # their weights
        self.z_mean = nnx.Variable(jnp.zeros((self.lat_dim,), jnp.float32))
        self.z_std = nnx.Variable(jnp.ones((self.lat_dim,), jnp.float32))
        self.G_mean = nnx.Variable(jnp.zeros((), jnp.float32))
        self.G_std = nnx.Variable(jnp.ones((), jnp.float32))
        self.E_mean = nnx.Variable(jnp.zeros((), jnp.float32))
        self.E_std = nnx.Variable(jnp.ones((), jnp.float32))
        self.fitted = nnx.Variable(jnp.zeros((3,), jnp.float32))                # [G, E, z] heads trained
        self.intrinsic_dim = nnx.Variable(jnp.zeros((), jnp.float32))
        self.latent_space = self.G = self.F = self.mu = self.B = self.F_mean = self.F_std = nnx.data(None)
        self._allocate(n_particles, feat_dim, rngs)

        # Everything outside the state: hetsiren, k-NN searcher and jitted functions
        self._hetsiren = _Opaque()
        self._derived = _Opaque({})

    # ------------------------------------------------------------------ #
    # State allocation and derived objects                               #
    # ------------------------------------------------------------------ #
    def _allocate(self, n_particles=None, feat_dim=None, rngs=None):
        """Create the variables whose shapes depend on ``n_particles`` / ``feat_dim`` (once known)"""
        if n_particles is not None and self.latent_space is None:
            n = int(n_particles)
            self.config["n_particles"] = n
            self.latent_space = nnx.data(nnx.Variable(jnp.zeros((n, self.lat_dim), jnp.float32)))
            self.G = nnx.data(nnx.Variable(jnp.zeros((n,), jnp.float32)))
        if feat_dim is not None and self.F is None:
            d, n = int(feat_dim), self.config["n_particles"]
            if n is None:
                raise RuntimeError("feat_dim given before n_particles")
            self.config["feat_dim"] = d
            self.F = nnx.data(nnx.Variable(jnp.zeros((n, d), jnp.float32)))
            self.mu = nnx.data(nnx.Variable(jnp.zeros((3 * self.max_points,), jnp.float32)))
            self.B = nnx.data(nnx.Variable(jnp.zeros((3 * self.max_points, d), jnp.float32)))
            self.F_mean = nnx.data(nnx.Variable(jnp.zeros((d,), jnp.float32)))
            self.F_std = nnx.data(nnx.Variable(jnp.ones((d,), jnp.float32)))
            if rngs is None:
                rngs = nnx.Rngs(random.randint(0, 2 ** 32 - 1))
            self.E_model = nnx.data(self._new_energy_head(d, rngs))

    def _new_energy_head(self, in_dim, rngs):
        return EnergyHead(in_dim, hidden=self.config["hidden_energy"], n_layers=self.config["n_layers_energy"],
                          n_wall_centres=self.config["n_wall_centres"], rngs=rngs)

    @property
    def hetsiren(self):
        if self._hetsiren.value is None:
            raise RuntimeError("No HetSIREN attached: call attach_model(graphdef, state) first")
        return self._hetsiren.value

    def attach_model(self, graphdef, state, latent_space=None):
        """Attach the HetSIREN (graphdef, state) and optionally the latent space of every particle"""
        self._hetsiren.value = (graphdef, state)
        if latent_space is not None:
            derived = {}
            latent_space = np.asarray(latent_space, np.float32)
            self._allocate(latent_space.shape[0])
            self.latent_space[...] = jnp.asarray(latent_space)
            latent_nn_fn, latent_nn_idx_fn = self._build_searcher_latent()
            derived["latent_nn_fn"] = latent_nn_fn
            derived["latent_nn_idx_fn"] = latent_nn_idx_fn
            self._derived.value = derived
        self.finalize()

    def finalize(self):
        """Rebuild every object derived from the variables"""
        derived = self._derived.value
        if self.F is not None and self._hetsiren.value is not None:
            derived["feature_fn"] = self._build_feature_fn()
        if self.F is not None:
            derived["feature_nn_fn"], derived["feature_nn_idx_fn"] = self._build_searcher_features()
        fitted = np.asarray(self.fitted[...]) > 0
        if fitted[self._FIT_G]:
            derived["G_fn"], derived["G_outside_fn"] = self._build_head_fn(
                self.G_model, self.z_mean[...], self.z_std[...], self.G_mean[...], self.G_std[...])
        if fitted[self._FIT_E] and self.E_model is not None:
            derived["E_fn"], derived["E_outside_fn"] = self._build_head_fn(
                self.E_model, self.F_mean[...], self.F_std[...], self.E_mean[...], self.E_std[...])
        self._derived.value = derived

    def _get_derived(self, name, needs):
        fn = self._derived.value.get(name)
        if fn is None:
            raise RuntimeError(f"Call {needs}() first")
        return fn

    @property
    def feature_fn(self):
        return self._get_derived("feature_fn", "prepare_space_cost")

    @property
    def feature_nn_fn(self):
        return self._get_derived("feature_nn_fn", "prepare_space_cost")

    @property
    def feature_nn_idx_fn(self):
        return self._get_derived("feature_nn_idx_fn", "prepare_space_cost")

    @property
    def latent_nn_fn(self):
        return self._get_derived("latent_nn_fn", "attach_model")

    @property
    def latent_nn_idx_fn(self):
        return self._get_derived("latent_nn_idx_fn", "attach_model")

    @property
    def G_fn(self):
        return self._get_derived("G_fn", "free_energy_function")

    @property
    def G_outside_fn(self):
        return self._get_derived("G_outside_fn", "free_energy_function")

    @property
    def E_fn(self):
        return self._get_derived("E_fn", "energy_from_features_function")

    @property
    def E_outside_fn(self):
        return self._get_derived("E_outside_fn", "energy_from_features_function")

    def _decode_gaussians_fn(self):
        graphdef, state = self.hetsiren

        @jax.jit
        def decode_gaussians(z):
            model = nnx.merge(graphdef, state)
            coords, values = model.delta_volume_decoder(z)
            return coords, values

        return decode_gaussians

    def _build_feature_fn(self):
        """Differentiable map z -> F"""
        decode_gaussians = self._decode_gaussians_fn()
        idx_j, w_j, mu_j, B_j = self.idx[...], self.w[...], self.mu[...], self.B[...]

        @jax.jit
        def project_feature_in_basis(z):
            # TODO: Compute covariance accumulative and use here mahalanobis distance
            coords = decode_gaussians(z)[0][:, idx_j]
            raw = (coords * w_j[:, None]).reshape(len(z), -1)
            return (raw - mu_j) @ B_j

        return project_feature_in_basis

    def _build_searcher_features(self):
        """Nearest-neighbour index on F. Returns (distances_fn, indices_fn)"""
        print(f"{bcolors.OKCYAN}\n###### Building nearest-neighbour index on feature space... ######")
        # np.array (not asarray): a NumPy view of a JAX buffer is read-only, and numba types read-only
        # arrays differently, which makes pynndescent's jitted tree builder fail type inference
        F = np.array(self.F[...])
        if jax.default_backend() == "cpu":
            searcher = pynndescent.NNDescent(F)
            searcher.prepare()
            feature_nn_fn = lambda x: searcher.query(x, k=self.k)[1]
            feature_nn_idx_fn = lambda x: searcher.query(x, k=self.k)[0]
        elif jax.default_backend() == "gpu":
            searcher = NearestNeighbors(n_neighbors=self.k)
            searcher.fit(F)
            feature_nn_fn = lambda x: searcher.kneighbors(x)[0]
            feature_nn_idx_fn = lambda x: searcher.kneighbors(x)[1]
        else:
            raise ValueError(f"Backend {jax.default_backend()} not supported")
        return feature_nn_fn, feature_nn_idx_fn

    def _build_searcher_latent(self):
        """Nearest-neighbour index on Z. Returns (distances_fn, indices_fn)"""
        print(f"{bcolors.OKCYAN}\n###### Building nearest-neighbour index on latent space... ######")
        # np.array (not asarray): a NumPy view of a JAX buffer is read-only, and numba types read-only
        # arrays differently, which makes pynndescent's jitted tree builder fail type inference
        latent_space = np.array(self.latent_space[...])
        if jax.default_backend() == "cpu":
            searcher = pynndescent.NNDescent(latent_space)
            searcher.prepare()
            latent_nn_fn = lambda x: searcher.query(x, k=self.k)[1]
            latent_nn_idx_fn = lambda x: searcher.query(x, k=self.k)[0]
        elif jax.default_backend() == "gpu":
            searcher = NearestNeighbors(n_neighbors=self.k)
            searcher.fit(latent_space)
            latent_nn_fn = lambda x: searcher.kneighbors(x)[0]
            latent_nn_idx_fn = lambda x: searcher.kneighbors(x)[1]
        else:
            raise ValueError(f"Backend {jax.default_backend()} not supported")
        return latent_nn_fn, latent_nn_idx_fn

    @staticmethod
    def _build_head_fn(model, x_mean, x_std, y_mean, y_std):
        """Jitted (energy_fn, outside_fn) for an EnergyHead, in raw input / energy units"""
        graphdef, state = nnx.split(model)

        @jax.jit
        def energy_fn(x):
            m = nnx.merge(graphdef, state)
            x = (jnp.asarray(x, jnp.float32) - x_mean) / x_std
            return m(x) * y_std + y_mean

        @jax.jit
        def outside_fn(x):
            m = nnx.merge(graphdef, state)
            x = (jnp.asarray(x, jnp.float32) - x_mean) / x_std
            return m.wall(x)

        return energy_fn, outside_fn

    # ------------------------------------------------------------------ #
    # Pipeline                                                           #
    # ------------------------------------------------------------------ #
    def prepare_free_energy_function(self, epochs=200, batch_size=1024, learning_rate=1e-3, grad_penalty=1e-3,
                                     val_fraction=0.1, log_dir=None, features_energy=True):
        self.prepare_space_cost()
        self.free_energy()
        self.free_energy_function(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                  grad_penalty=grad_penalty, val_fraction=val_fraction, log_dir=log_dir)
        if features_energy:
            self.energy_from_features_function(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                               grad_penalty=grad_penalty, val_fraction=val_fraction,
                                               log_dir=log_dir)

    def compute_free_energy_at(self, z):
        return self.G_fn(z)

    def compute_energy_from_features_at(self, F):
        return self.E_fn(F)

    def outside_latent(self, z):
        """Wall score in [0, 1]: ~0 inside the latent data, ~1 where G saturates"""
        return self.G_outside_fn(z)

    def outside_features(self, F):
        """Wall score in [0, 1]: ~0 inside the feature data, ~1 where E saturates"""
        return self.E_outside_fn(F)

    def prepare_space_cost(self, batch_size=64):
        from hax.generators import NumpyGenerator

        if self.latent_space is None:
            raise RuntimeError("Call attach_model(graphdef, state, latent_space) first")
        latent_space = np.asarray(self.latent_space[...])
        max_points = self.max_points

        # Prepare data loader
        data_loader = NumpyGenerator(latent_space).return_grain_dataset(batch_size=batch_size,
                                                                        shuffle=False, preShuffle=False,
                                                                        num_epochs=None, num_workers=0)
        steps_per_epoch = max(1, int(latent_space.shape[0] / batch_size))

        # Jitted functions
        decode_gaussians = self._decode_gaussians_fn()

        @jax.jit
        def raw_gaussian_features(z):
            # TODO: Compute covariance accumulative and use here mahalanobis distance
            coords = decode_gaussians(z)[0][:, idx_j]
            return (coords * w_j[:, None]).reshape(len(z), -1)

        # Get Gaussian statistics to compute the variable mobile Gaussians
        print(f"{bcolors.OKCYAN}\n###### Computing Gaussian statistics... ######")
        mean_a = mean_mu = mean_mu_sqr = None
        with closing(iter(data_loader)) as iter_data_loader:
            for _ in _pbar(range(steps_per_epoch)):
                (z, _) = next(iter_data_loader)
                coords, values = decode_gaussians(z)

                if mean_a is None:
                    N = coords.shape[1]
                    mean_a, mean_mu, mean_mu_sqr = np.zeros(N), np.zeros((N, 3)), np.zeros((N, 3))

                mean_a  += values.sum(axis=0)
                mean_mu  += coords.sum(axis=0)
                mean_mu_sqr += (coords ** 2).sum(axis=0)
        mean_a = mean_a / steps_per_epoch
        spread = np.sqrt(np.maximum(mean_mu_sqr / steps_per_epoch - (mean_mu / steps_per_epoch ) ** 2, 0)).sum(axis=-1)
        idx = np.sort(np.argsort(-spread)[:max_points])  # Most variable Gaussian indices
        w = mean_a[idx]
        w = (w / np.linalg.norm(w)).astype(np.float32)
        idx_j, w_j = jnp.asarray(idx, jnp.int32), jnp.asarray(w)

        # PCA basis on the Gaussians: compute feature to train the basis
        print(f"{bcolors.OKCYAN}\n###### Computing Gaussian features... ######")
        features = np.empty((len(latent_space), 3 * max_points), np.float32)
        with closing(iter(data_loader)) as iter_data_loader:
            for _ in _pbar(range(steps_per_epoch)):
                (z, idx_batch) = next(iter_data_loader)
                features[idx_batch] = np.asarray(raw_gaussian_features(z))

        # Train PCA basis on the features and get the mean and basis vectors
        print(f"{bcolors.OKCYAN}\n###### Fitting PCA basis on Gaussian features... ######")
        pca = PCA(n_components=96, svd_solver='randomized', copy=False)
        pca.fit(features)
        var = np.cumsum(pca.explained_variance_ratio_)
        n_comp = int(np.searchsorted(var, 0.98)) + 1
        if n_comp >= len(var):
            print(f"WARNING: {len(var)} components reach only {var[-1]:.3f} "
                f"variance -- raise n_comp_max")
        mu = pca.mean_.astype(np.float32)
        B = np.ascontiguousarray(pca.components_[:n_comp].T, dtype=np.float32)

        # Store the Gaussian selection and basis, then build the differentiable map z -> F
        self._allocate(feat_dim=n_comp)
        self.idx[...], self.w[...] = idx_j, w_j
        self.mu[...], self.B[...] = jnp.asarray(mu), jnp.asarray(B)
        feature_fn = self._build_feature_fn()

        # Build feature space
        print(f"{bcolors.OKCYAN}\n###### Projecting features onto PCA basis ({n_comp} components)... ######")
        F = np.zeros((len(latent_space), n_comp), np.float32)
        with closing(iter(data_loader)) as iter_data_loader:
            for _ in _pbar(range(steps_per_epoch)):
                (z, idx_batch) = next(iter_data_loader)
                F[idx_batch] = np.asarray(feature_fn(z))
        self.F[...] = jnp.asarray(F)

        # Nearest neighbour on F
        feature_nn_fn, feature_nn_idx_fn = self._build_searcher_features()
        self._derived.value.update(feature_fn=feature_fn, feature_nn_fn=feature_nn_fn, feature_nn_idx_fn=feature_nn_idx_fn)

    def intrinsic_dimension(self, n_sample=20_000, discard=0.10):
        # TwoNN estimator (Facco et al. 2017)
        F = np.asarray(self.F[...])
        s = np.random.choice(len(F), min(n_sample, len(F)), replace=False)
        d = self.feature_nn_fn(F[s])
        r1, r2 = d[:, 1], d[:, 2]

        ok = (r1 > 0) & np.isfinite(r2)
        mu = np.sort(r2[ok] / r1[ok])
        mu = mu[mu > 1.0]
        mu = mu[:int(len(mu) * (1 - discard))]

        Fcdf = np.arange(1, len(mu) + 1) / (len(mu) + 1)
        x, y = np.log(mu), -np.log1p(-Fcdf)
        self.intrinsic_dim[...] = jnp.asarray(float((x @ y) / (x @ x)), jnp.float32)
        return float(self.intrinsic_dim[...])

    def free_energy(self, kT=1.0, intrinsic_dim=None, batch_size=64):
        from hax.generators import NumpyGenerator  # lazy: hax.generators imports hax.utils

        # Compute intrinsic dimension to correct the free energy
        if intrinsic_dim is None:
            intrinsic_dim = self.intrinsic_dimension()
        else:
            self.intrinsic_dim[...] = jnp.asarray(float(intrinsic_dim), jnp.float32)
        F = np.asarray(self.F[...])

        # Prepare data loader
        data_loader = NumpyGenerator(F).return_grain_dataset(batch_size=batch_size,
                                                             shuffle=False, preShuffle=False,
                                                             num_epochs=None, num_workers=0)
        steps_per_epoch = max(1, int(F.shape[0] / batch_size))

        # Get distaces to furthes neighbors
        print(f"{bcolors.OKCYAN}\n###### Computing k-NN free energy... ######")
        r_k = np.empty(len(F), np.float64)
        with closing(iter(data_loader)) as iter_data_loader:
            for _ in _pbar(range(steps_per_epoch)):
                (F_batch, idx) = next(iter_data_loader)
                d = self.feature_nn_fn(F_batch)
                r_k[idx] = d[:, -1]

        if (r_k <= 0).any():
            n_dup = int((r_k <= 0).sum())
            print(f"WARNING: {n_dup} particles have zero k-NN radius (duplicates)")
            r_k = np.maximum(r_k, np.median(r_k) * 1e-6)

        # log of the unit-ball volume in d_eff dimensions
        log_V = 0.5 * intrinsic_dim * np.log(np.pi) - gammaln(0.5 * intrinsic_dim + 1)
        logp = np.log(self.k) - np.log(len(F)) - log_V - intrinsic_dim * np.log(r_k)

        # Compute free energy
        G = -kT * logp
        self.G[...] = jnp.asarray(G - np.percentile(G, 1), jnp.float32)

    # ------------------------------------------------------------------ #
    # Small-network training helpers                                     #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _split_train_val(n, val_fraction, batch_size):
        perm = np.random.permutation(n)
        n_val = int(n * val_fraction)
        val_idx, train_idx = perm[:n_val], perm[n_val:]
        batch_size = min(batch_size, len(train_idx))
        return train_idx, val_idx, batch_size

    @staticmethod
    def _fit_loop(name, epochs, steps_per_epoch, epoch_fn, val_fn, model, writer, tags, patience, min_delta):
        key = jax.random.PRNGKey(random.randint(0, 2 ** 32 - 1))
        best_val, best_state, best_epoch, stale = np.inf, None, 0, 0
        val = None

        print(f"{bcolors.OKCYAN}\n###### Training {name}... ######")
        pbar = _pbar(range(epochs))
        for epoch in pbar:
            pbar.set_description(f"Epoch {epoch + 1}/{epochs}")
            train = {k: float(v) for k, v in epoch_fn(jax.random.fold_in(key, epoch)).items()}
            step = (epoch + 1) * steps_per_epoch

            if val_fn is not None:
                val = {k: float(v) for k, v in val_fn().items()}
                metric = val[next(iter(tags))]
                if metric < best_val - min_delta:
                    best_val, best_epoch, stale = metric, epoch, 0
                    best_state = jax.tree.map(lambda x: x, nnx.state(model, nnx.Param))
                else:
                    stale += 1

            if writer is not None:
                for k, tag in tags.items():
                    scalars = {"train": train[k]}
                    if val is not None:
                        scalars["validation"] = val[k]
                    writer.add_scalars(tag, scalars, step)

            postfix = " | ".join(f"{k}={v:.5f}" for k, v in train.items())
            if val is not None:
                postfix += " | " + " | ".join(f"val_{k}={v:.5f}" for k, v in val.items())
            pbar.set_postfix_str(postfix)

            if val_fn is not None and stale >= patience:
                print(f"\nEarly stop at epoch {epoch + 1}: no validation improvement for {patience} epochs "
                      f"(best epoch {best_epoch + 1})")
                break

        if best_state is not None:
            nnx.update(model, best_state)

    def _fit_energy_head(self, name, tag, X_all, Y_all, epochs, batch_size, learning_rate, weight_decay,
                         grad_penalty, val_fraction, clip_percentiles, patience, min_delta, log_dir,
                         wall, wall_height, wall_radius_percentile, wall_radius_margin, neg_weight, neg_sigma):
        """Fit an EnergyHead Y = f(X). Returns (model, x_mean, x_std, y_mean, y_std)"""
        from hax.metrics import JaxSummaryWriter

        X_all = np.asarray(X_all, np.float32)
        Y_all = np.asarray(Y_all, np.float32)
        n, in_dim = X_all.shape

        # Robust target: clip the tails, then standardise (kept to map predictions back to energy units)
        if clip_percentiles is not None:
            lo, hi = np.percentile(Y_all, clip_percentiles)
            Y_all = np.clip(Y_all, lo, hi)
        x_mean, x_std = X_all.mean(0), X_all.std(0) + 1e-8
        y_mean, y_std = Y_all.mean(), Y_all.std() + 1e-8
        X_all = (X_all - x_mean) / x_std
        Y_all = (Y_all - y_mean) / y_std

        # Fresh model, wall (fitted, not trained) and saturation height in standardised units
        rng_key = jax.random.PRNGKey(random.randint(0, 2 ** 32 - 1))
        model = self._new_energy_head(in_dim, nnx.Rngs(rng_key))
        if wall_height is None:
            height = float(Y_all.max() + 2.0 * (Y_all.max() - Y_all.min()))
        else:
            height = float((wall_height - y_mean) / y_std)
        model.height[...] = jnp.asarray(height, jnp.float32)
        if wall:
            print(f"{bcolors.OKCYAN}\n###### Fitting support wall for {name}... ######")
            model.wall.fit(X_all, radius_percentile=wall_radius_percentile, radius_margin=wall_radius_margin)
        else:
            neg_weight = 0.0
        use_negatives = neg_weight > 0

        # Train / validation split (device resident)
        train_idx, val_idx, batch_size = self._split_train_val(n, val_fraction, batch_size)
        n_train, n_val = len(train_idx), len(val_idx)
        steps_per_epoch = max(1, n_train // batch_size)
        X_train, Y_train = jnp.asarray(X_all[train_idx]), jnp.asarray(Y_all[train_idx])
        X_val, Y_val = jnp.asarray(X_all[val_idx]), jnp.asarray(Y_all[val_idx])

        schedule = optax.cosine_decay_schedule(learning_rate, epochs * steps_per_epoch)
        optimizer = nnx.Optimizer(model, optax.adamw(schedule, weight_decay=weight_decay), wrt=nnx.Param)
        writer = JaxSummaryWriter(log_dir) if log_dir is not None else None

        def loss_fn(model, x, y, key):
            pred = model(x)
            mse = jnp.mean((pred - y) ** 2)

            # Smoothness: discourage steep local slopes of the fitted landscape
            dG = jax.vmap(jax.grad(lambda xi: model(xi[None])[0]))(x)
            smooth = jnp.mean(jnp.sum(dG ** 2, axis=-1))
            loss = mse + grad_penalty * smooth

            if use_negatives:
                # Negatives: batch points displaced by neg_sigma local radii (Gaussian norm ~ sigma * sqrt(dim)).
                # Only those the wall reports as outside count, and the raw MLP must reach the wall height there
                k_sigma, k_noise = jax.random.split(key)
                sigma = jax.random.uniform(k_sigma, (x.shape[0], 1), minval=neg_sigma[0], maxval=neg_sigma[1])
                scale = sigma * model.wall.local_radius(x)[:, None] / jnp.sqrt(x.shape[1])
                x_neg = x + scale * jax.random.normal(k_noise, x.shape)
                outside = jax.lax.stop_gradient(model.wall(x_neg) > 0.5).astype(jnp.float32)
                hinge = jnp.sum(outside * jax.nn.relu(model.height[...] - model.mlp(x_neg)))
                loss = loss + neg_weight * hinge / (jnp.sum(outside) + 1.0)

            return loss, mse

        @nnx.jit
        def train_epoch(model, optimizer, key):
            k_perm, k_neg = jax.random.split(key)
            batches = jax.random.permutation(k_perm, n_train)[:steps_per_epoch * batch_size]
            batches = batches.reshape(steps_per_epoch, batch_size)
            keys = jax.random.split(k_neg, steps_per_epoch)

            # Model and optimizer travel in the carry: as broadcast arguments their updates are dropped
            @nnx.scan(in_axes=(nnx.Carry, 0, 0), out_axes=(nnx.Carry, 0))
            def body(carry, idx, k):
                total, model, optimizer = carry
                (loss, mse), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model, X_train[idx], Y_train[idx], k)
                optimizer.update(model, grads)
                return (total + loss, model, optimizer), mse

            (total_loss, _, _), mses = body((0.0, model, optimizer), batches, keys)
            return {"loss": total_loss / steps_per_epoch, "mse": jnp.mean(mses)}

        @nnx.jit
        def validation(model):
            mse = jnp.mean((model(X_val) - Y_val) ** 2)
            return {"mse": mse, "loss": mse}

        self._fit_loop(name, epochs, steps_per_epoch,
                       lambda key: train_epoch(model, optimizer, key),
                       (lambda: validation(model)) if n_val > 0 else None,
                       model, writer,
                       {"mse": f"Fit MSE ({tag})", "loss": f"Training loss ({tag})"},
                       patience, min_delta)

        if writer is not None:
            writer.close()

        model.eval()
        return model, x_mean, x_std, y_mean, y_std

    def free_energy_function(self, epochs=200, batch_size=1024,
                             learning_rate=1e-3, weight_decay=1e-4, grad_penalty=1e-3,
                             val_fraction=0.1, clip_percentiles=(1.0, 99.0), patience=15, min_delta=1e-4,
                             log_dir=None, wall=True, wall_height=None, wall_radius_percentile=98.0,
                             wall_radius_margin=1.25, neg_weight=0.1, neg_sigma=(1.0, 3.0)):
        """Fit a small MLP so that G = f(z), saturating to ``wall_height`` outside the latent data"""
        if self.G is None or float(jnp.sum(jnp.abs(self.G[...]))) == 0.0:
            raise RuntimeError("Call free_energy() before free_energy_function()")

        model, z_mean, z_std, G_mean, G_std = self._fit_energy_head(
            "free energy function", "FreeEnergyMLP z", self.latent_space[...], self.G[...],
            epochs, batch_size, learning_rate, weight_decay, grad_penalty, val_fraction, clip_percentiles,
            patience, min_delta, log_dir, wall, wall_height, wall_radius_percentile, wall_radius_margin,
            neg_weight, neg_sigma)

        self.G_model = model
        self.z_mean[...], self.z_std[...] = jnp.asarray(z_mean, jnp.float32), jnp.asarray(z_std, jnp.float32)
        self.G_mean[...], self.G_std[...] = jnp.asarray(G_mean, jnp.float32), jnp.asarray(G_std, jnp.float32)
        self.fitted[...] = self.fitted[...].at[self._FIT_G].set(1.0)
        self._derived.value["G_fn"], self._derived.value["G_outside_fn"] = self._build_head_fn(
            model, self.z_mean[...], self.z_std[...], self.G_mean[...], self.G_std[...])

    def energy_from_features_function(self, epochs=200, batch_size=1024,
                                      learning_rate=1e-3, weight_decay=1e-4, grad_penalty=1e-3,
                                      val_fraction=0.1, clip_percentiles=(1.0, 99.0), patience=15, min_delta=1e-4,
                                      log_dir=None, wall=True, wall_height=None, wall_radius_percentile=98.0,
                                      wall_radius_margin=1.25, neg_weight=0.1, neg_sigma=(1.0, 3.0)):
        """Fit a small MLP so that E = f(F): the free energy predicted from the Gaussian features directly,
        saturating to ``wall_height`` outside the feature data"""
        if self.F is None:
            raise RuntimeError("Call prepare_space_cost() before energy_from_features_function()")
        if self.G is None or float(jnp.sum(jnp.abs(self.G[...]))) == 0.0:
            raise RuntimeError("Call free_energy() before energy_from_features_function()")

        model, F_mean, F_std, E_mean, E_std = self._fit_energy_head(
            "feature energy function", "FreeEnergyMLP F", self.F[...], self.G[...],
            epochs, batch_size, learning_rate, weight_decay, grad_penalty, val_fraction, clip_percentiles,
            patience, min_delta, log_dir, wall, wall_height, wall_radius_percentile, wall_radius_margin,
            neg_weight, neg_sigma)

        self.E_model = model
        self.F_mean[...], self.F_std[...] = jnp.asarray(F_mean, jnp.float32), jnp.asarray(F_std, jnp.float32)
        self.E_mean[...], self.E_std[...] = jnp.asarray(E_mean, jnp.float32), jnp.asarray(E_std, jnp.float32)
        self.fitted[...] = self.fitted[...].at[self._FIT_E].set(1.0)
        self._derived.value["E_fn"], self._derived.value["E_outside_fn"] = self._build_head_fn(
            model, self.F_mean[...], self.F_std[...], self.E_mean[...], self.E_std[...])

    def features_at_latent(self, z):
        """Features of arbitrary latents through the exact, differentiable z -> F map (no particle snapping)"""
        return self.feature_fn(jnp.asarray(z, jnp.float32))

    @partial(jax.jit, static_argnums=(0,), static_argnames=("max_nodes", "max_iters", "batch", "adaptive_bounding_box"))
    def _loop_trrt(self, key, features_lo, features_hi, features_init, features_goal, *, max_nodes, max_iters,
                   batch, step, goal_tol, goal_bias, T0, T_rate, rho, cmax, max_T_step, T_max, adaptive_bounding_box,
                   box_margin):

        # Initialization
        d = self.config["feat_dim"]
        N = max_nodes
        B = batch

        nodes = jnp.zeros((N + 1, d)).at[0].set(features_init)
        parent = jnp.full(N + 1, -1, jnp.int32)
        cost_value = jnp.full(N + 1, jnp.inf).at[0].set(self.compute_energy_from_features_at(features_init[None])[0])
        idx_all = jnp.arange(N + 1)

        state = dict(nodes=nodes, parent=parent, cost_value=cost_value, count=jnp.int32(1),
                     T=jnp.float32(T0), n_refined=jnp.int32(0),
                     key=key, it=jnp.int32(0), goal_idx=jnp.int32(-1))

        # Condition function for Jax while loop
        def cond(s):
            return (s["goal_idx"] < 0) & (s["it"] < max_iters) & (s["count"] < N)

        def body(s):
            # Initialization
            key, k1, k2 = jax.random.split(s["key"], 3)
            nodes, count = s["nodes"], s["count"]

            # T-RRT in batches Step 1: Get B Q_rand points
            if adaptive_bounding_box:
                filled = (idx_all < count)[:, None]
                nodes_lo = jnp.min(jnp.where(filled, nodes, jnp.inf), axis=0) - box_margin * step
                nodes_hi = jnp.max(jnp.where(filled, nodes, -jnp.inf), axis=0) + box_margin * step
                nodes_lo = jnp.maximum(nodes_lo, features_lo)
                nodes_hi = jnp.minimum(nodes_hi, features_hi)
                Q_rand = nodes_lo + (nodes_hi - nodes_lo) * jax.random.uniform(k1, (B, d))
            else:
                Q_rand = features_lo + (features_hi - features_lo) * jax.random.uniform(k1, (B, d))
            use_goal = jax.random.uniform(k2, (B,)) < goal_bias
            Q_rand = jnp.where(use_goal[:, None], features_goal[None], Q_rand)

            # Step 2: Find closest neighbrous in nodes to Q_rand
            p2 = jnp.sum(nodes ** 2, -1)
            d2 = jnp.sum(Q_rand ** 2, -1)[:, None] + p2[None, :] - 2.0 * Q_rand @ nodes.T
            d2 = jnp.where((idx_all < count)[None, :], d2, jnp.inf)
            near = jnp.argmin(d2, axis=1)
            Q_near = nodes[near]

            # Step 3: Interpolate along the segment Q_near -> Q_rand, with a maximum step size of step
            diff = Q_rand - Q_near
            dist = jnp.linalg.norm(diff, axis=-1)
            refining = dist < step
            Q_new = jnp.where(refining[:, None], Q_rand,
                              Q_near + step * diff / jnp.maximum(dist, 1e-9)[:, None])
            
            # Minimal expansion control to prevent  graph to accept only points not expanding the graph
            refine_ok = ~(refining & (s["n_refined"] > rho * count))

            # Step 3: Transition test: downhill always passes; uphill passes when exp(-(c_j - c_i) / T) > 0.5
            cost_near = s["cost_value"][near]
            cost_new = self.compute_energy_from_features_at(Q_new)
            dc = cost_new - cost_near
            downhill = dc <= 0.0
            uphill_ok = jnp.exp(-dc / s["T"]) > 0.5
            trans_ok = (downhill | uphill_ok) & (cost_new <= cmax)
            tested = refine_ok & ~downhill
            accept = refine_ok & trans_ok

            # Step 4: Temperature update
            filled = idx_all < count
            cost_range = jnp.max(jnp.where(filled, s["cost_value"], -jnp.inf)) \
                - jnp.min(jnp.where(filled, s["cost_value"], jnp.inf))
            cost_range = jnp.maximum(cost_range, 1e-9)
            cooling = jnp.sum(jnp.where(tested & trans_ok, dc, 0.0)) / cost_range
            heating = jnp.sum(tested & ~trans_ok).astype(jnp.float32) * T_rate
            net = jnp.clip(heating - cooling, -max_T_step, max_T_step)
            T = jnp.clip(s["T"] * 2.0 ** net, 1e-12, T_max)
            
            # Step 5: Update graph
            pos = jnp.cumsum(accept) - 1
            accept = accept & (count + pos < N)
            slot = jnp.where(accept, count + pos, N)
            nodes = nodes.at[slot].set(Q_new)
            parent = s["parent"].at[slot].set(near)
            cost_value = s["cost_value"].at[slot].set(cost_new)
            n_new = jnp.sum(accept).astype(jnp.int32)

            # --- goal check: first accepted sample inside the goal ball
            hit = accept & (jnp.linalg.norm(Q_new - features_goal, axis=-1) < goal_tol)
            first = jnp.argmax(hit)
            goal_idx = jnp.where(jnp.any(hit), slot[first], s["goal_idx"])
            return dict(nodes=nodes, parent=parent, cost_value=cost_value, count=count + n_new, T=T,
                        n_refined=s["n_refined"] + jnp.sum(accept & refining).astype(jnp.int32),
                        key=key, it=s["it"] + 1, goal_idx=goal_idx)

        s = jax.lax.while_loop(cond, body, state)
        return s["nodes"][:N], s["parent"][:N], s["cost_value"][:N], s["count"], s["goal_idx"], s["it"], s["T"]

    def feature_energies(self, batch_size=4096):
        """Feature energy head evaluated on every training particle (raw units), in batches"""
        F = np.asarray(self.F[...], np.float32)
        return np.concatenate([np.asarray(self.compute_energy_from_features_at(F[start:start + batch_size]))
                               for start in range(0, len(F), batch_size)])

    def _feature_energy_stats(self, ceiling_percentile=99.0, floor_percentile=1.0, batch_size=4096):
        """(floor, ceiling) of the feature energy head over the training particles as percentiles)"""
        E = self.feature_energies(batch_size)
        return float(np.percentile(E, floor_percentile)), float(np.percentile(E, ceiling_percentile))

    def find_basins(self, n_basins=2, min_separation=None, labels=None, cluster=False, energies=None,
                    smooth=True, batch_size=4096):
        """Find particles at the bottom of the energy basins"""
        F = np.asarray(self.F[...], np.float32)
        E = self.feature_energies(batch_size) if energies is None else np.asarray(energies, np.float32)
        if E.shape[0] != len(F):
            raise ValueError(f"energies has {E.shape[0]} entries, expected {len(F)}")

        # Neighbourhoods in feature space (self in column 0)
        nn = np.empty((len(F), self.k), np.int64)
        for start in range(0, len(F), batch_size):
            nn[start:start + batch_size] = np.asarray(self.feature_nn_idx_fn(F[start:start + batch_size]))
        if smooth:
            E = E[nn].mean(axis=1)

        # Local minima: energy not above that of any neighbour
        is_min = E <= E[nn[:, 1:]].min(axis=1)
        if not is_min.any():
            raise RuntimeError("no local minima found in the feature energy landscape")

        if labels is None and cluster:
            from sklearn.cluster import MiniBatchKMeans
            labels = MiniBatchKMeans(n_clusters=n_basins, n_init=3, batch_size=4096).fit(F).labels_

        if labels is not None:
            # One basin per cluster: its deepest local minimum (deepest particle if the cluster has none)
            labels = np.asarray(labels)
            if labels.shape[0] != len(F):
                raise ValueError(f"labels has {labels.shape[0]} entries, expected {len(F)}")
            chosen, chosen_labels = [], []
            for lab in np.unique(labels):
                members = np.flatnonzero((labels == lab) & is_min)
                if len(members) == 0:
                    members = np.flatnonzero(labels == lab)
                chosen.append(int(members[np.argmin(E[members])]))
                chosen_labels.append(lab)
            order = np.argsort(E[chosen])
            chosen = np.asarray(chosen, np.int64)[order]
            return chosen, E[chosen], np.asarray(chosen_labels)[order]

        # No clusters: greedy by depth with a separation constraint
        minima = np.flatnonzero(is_min)
        if min_separation is None:
            min_separation = 0.1 * float(np.linalg.norm(F.max(0) - F.min(0)))
        chosen = []
        for i in minima[np.argsort(E[minima])]:
            if all(np.linalg.norm(F[i] - F[j]) >= min_separation for j in chosen):
                chosen.append(int(i))
            if len(chosen) == n_basins:
                break
        if len(chosen) < n_basins:
            print(f"WARNING: only {len(chosen)} basins found at separation {min_separation:.3g} "
                  f"(from {len(minima)} local minima); lower min_separation")
        chosen = np.asarray(chosen, np.int64)
        return chosen, E[chosen], None

    def _path_profile(self, path, E_floor, quad):
        d = self.config["feat_dim"]
        pj = jnp.asarray(path, jnp.float32)
        t = (jnp.arange(quad) + 0.5) / quad
        u, v = pj[:-1], pj[1:]
        seg = u[:, None, :] + t[None, :, None] * (v - u)[:, None, :]
        fv = jnp.maximum(self.compute_energy_from_features_at(seg.reshape(-1, d)).reshape(-1, quad).mean(-1)
                         - E_floor, 0.0)
        edge = jnp.linalg.norm(v - u, axis=-1) * fv
        cum = jnp.concatenate([jnp.zeros(1), jnp.cumsum(edge)])
        energy = self.compute_energy_from_features_at(pj) - E_floor
        return np.asarray(energy), np.asarray(cum)

    def _path_to_latent(self, path, latent_init, latent_goal):
        particle_idx = np.asarray(self.feature_nn_idx_fn(np.ascontiguousarray(path, np.float32)))[:, 0]
        keep = np.concatenate([[True], particle_idx[1:] != particle_idx[:-1]])
        particle_idx = particle_idx[keep]
        path_latent = np.asarray(self.latent_space[...])[particle_idx]
        path_latent[0], path_latent[-1] = np.asarray(latent_init), np.asarray(latent_goal)
        return path_latent, particle_idx

    def plan_trrt(self, latent_init, latent_goal, *, max_nodes=20000, max_iters=20000, batch=128, step=None,
                  goal_tol=None, goal_bias=0.05, T0=1e-2, T_rate=0.1, rho=0.1, max_T_step=4, tau=None,
                  tau_percentile=99.0, quad=8, adaptive_bounding_box=True, box_margin=2.0):
        # Initialization: the query latents are mapped to FEATURE space, where the tree is grown
        latent_init = jnp.asarray(latent_init, jnp.float32).reshape(self.lat_dim)
        latent_goal = jnp.asarray(latent_goal, jnp.float32).reshape(self.lat_dim)
        features_init, features_goal = self.features_at_latent(jnp.stack([latent_init, latent_goal]))

        # Sampling bounds and step from the extent of the feature cloud
        F = self.F[...]
        features_lo, features_hi = F.min(0), F.max(0)
        diag = float(jnp.linalg.norm(features_hi - features_lo))
        step = diag * 0.02 if step is None else step

        goal_tol = step if goal_tol is None else goal_tol

        # Energy floor (1st percentile) and ceiling (tau_percentile) of the feature head over the particles
        E_floor, E_ceiling = self._feature_energy_stats(tau_percentile)
        tau = (E_ceiling - E_floor) if tau is None else float(tau)
        cmax = E_floor + tau

        # Temperature cap
        T_max = float(tau / np.log(2.0)) if np.isfinite(tau) else 1e6

        # T-RRT loop
        seed = jax.random.PRNGKey(random.randint(0, 2 ** 32 - 1))
        nodes, parent, cost_value, count, goal_idx, iters, T = self._loop_trrt(
        seed, features_lo, features_hi, features_init, features_goal,
        max_nodes=max_nodes, max_iters=max_iters, batch=batch, step=step, goal_tol=goal_tol,
        goal_bias=goal_bias, T0=T0, T_rate=T_rate, rho=rho, cmax=cmax, max_T_step=max_T_step, T_max=T_max,
        adaptive_bounding_box=adaptive_bounding_box, box_margin=box_margin)
        goal_idx = int(goal_idx)
        if goal_idx < 0:
            raise RuntimeError(f"no path after {int(iters)} batches ({int(iters)*batch} samples) / {int(count)} nodes; "
                            "raise max_iters/max_nodes, step, or T0")

        # Walk the tree back from the goal node to the root in feature space
        parent = np.asarray(parent)
        idx = [goal_idx]
        while idx[-1] != 0:
            idx.append(int(parent[idx[-1]]))
        idx = np.asarray(idx[::-1])
        path = np.asarray(nodes)[idx]
        path = np.concatenate([path, np.asarray(features_goal)[None]])

        energy, cum = self._path_profile(path, E_floor, quad)
        path_latent, particle_idx = self._path_to_latent(path, latent_init, latent_goal)

        info = dict(nodes=int(count), batches=int(iters), samples=int(iters) * batch, final_T=float(T),
                    particle_idx=particle_idx, path_features=path, energy_cum=np.asarray(cum),
                    energy_floor=E_floor, tau=tau, T_max=T_max)
        return path_latent, energy, cum, info

    @staticmethod
    def _chunked_map(fn, xs, chunk):
        leaves = jax.tree_util.tree_leaves(xs)
        n = leaves[0].shape[0]
        pad = (-n) % chunk

        def prep(x):
            x = jnp.pad(x, ((0, pad),) + ((0, 0),) * (x.ndim - 1))
            return x.reshape((-1, chunk) + x.shape[1:])

        out = jax.lax.map(fn, jax.tree_util.tree_map(prep, xs))
        return jax.tree_util.tree_map(lambda o: o.reshape((-1,) + o.shape[2:])[:n], out)

    @partial(jax.jit, static_argnums=(0,), static_argnames=("k", "chunk"))
    def _knn_graph_fmt(self, nodes, *, k, chunk):
        p2 = jnp.sum(nodes ** 2, -1)

        def one_chunk(Q):
            d2 = jnp.sum(Q ** 2, -1)[:, None] + p2[None, :] - 2.0 * Q @ nodes.T
            _, near = jax.lax.top_k(-d2, k + 1)
            return near[:, 1:]

        return self._chunked_map(one_chunk, nodes, chunk)

    @partial(jax.jit, static_argnums=(0,), static_argnames=("quad", "chunk"))
    def _edge_costs_fmt(self, nodes, near, E_floor, cmax, eps, *, quad, chunk):
        d = self.config["feat_dim"]
        k = near.shape[1]
        t = (jnp.arange(quad) + 0.5) / quad

        def one_chunk(args):
            # Step1: Get closest points Q_new to each node Q_near
            Q_near, nb = args
            Q_new = nodes[nb]

            # Step 2: Draw n samples (quad) along the segment Q_near -> Q_new
            diff = Q_new - Q_near[:, None, :]
            seg = Q_near[:, None, None, :] + t[None, None, :, None] * diff[:, :, None, :]

            # Step 3: Compute the cost value (energy) at the n samples along the segments Q_near -> Q_new
            fv = self.compute_energy_from_features_at(seg.reshape(-1, d)).reshape(-1, k, quad)

            # Step 4: Compute the distance of the segment Q_near -> Q_new
            dist = jnp.linalg.norm(diff, axis=-1)

            # Step 5: Cost of a an edge following the paper (average energy along the edge weighted by the 
            # edge distance)
            cost_edge = dist * jnp.maximum(fv.mean(-1) - E_floor, eps)

            return jnp.where(fv.max(-1) > cmax, jnp.inf, cost_edge)

        return self._chunked_map(one_chunk, (nodes, near), chunk)

    @staticmethod
    @partial(jax.jit, static_argnames=("n", "max_iters"))
    def _shortest_path_fmt(src_e, dst_e, w_e, n, max_iters):
        cost_value = jnp.full(n, jnp.inf).at[0].set(0.0)
        parent = jnp.full(n, -1, jnp.int32).at[0].set(0)

        def cond(s):
            return s[2] & (s[3] < max_iters)

        def body(s):
            cost_value, parent, _, it = s

            # Step 1: Accumulative cost going from Q_start to a given node through the best route
            cand = cost_value[src_e] + w_e

            # Step 2: For each destination node, get the minimum accumulative cost to reach it
            # (i.e. find segments ending in the node and get the minimum cost among them)
            best = jax.ops.segment_min(cand, dst_e, n)

            # Find which destination has an improved cost compared to its previous round
            # (i.e. if there is any cheaper route to reach a given node compared to previous iteration)
            improved = best < cost_value

            # Find the edge producing a winner destination and with improved cost over the 
            # previous iteration
            hit = (cand == best[dst_e]) & improved[dst_e]

            # Find the parent node connecting to the desination node whose edge became a winner based on 
            # prvious criteria (segment_max is basically used to ignore -1 entries representing source that 
            # whose edge did not lead to any improvement, ensuring that an arbitrary best node is chosen if 
            # there exists a tie)
            new_parent = jax.ops.segment_max(jnp.where(hit, src_e, -1), dst_e, n)

            # Update the cost value/parent list with the new best value (otherwise keep the previous 
            # iteration cost)
            cost_value = jnp.where(improved, best, cost_value)
            parent = jnp.where(improved, new_parent, parent)

            return cost_value, parent, jnp.any(improved), it + 1

        cost_value, parent, _, iters = jax.lax.while_loop(cond, body, (cost_value, parent, True, 0))
        return cost_value, parent, iters

    def plan_fmt(self, latent_init, latent_goal, *, n_samples=20000, k=None, quad=8, tau=None,
                 tau_percentile=99.0, eps=1e-3, chunk=512, max_iters=None, sample_particles=True):
        # Initialization
        latent_init = jnp.asarray(latent_init, jnp.float32).reshape(self.lat_dim)
        latent_goal = jnp.asarray(latent_goal, jnp.float32).reshape(self.lat_dim)
        features_init, features_goal = self.features_at_latent(jnp.stack([latent_init, latent_goal]))

        F = self.F[...]
        d = self.config["feat_dim"]
        n_samples = min(int(n_samples), F.shape[0])
        seed = jax.random.PRNGKey(random.randint(0, 2 ** 32 - 1))
        if sample_particles:
            Q_rand = F[jax.random.choice(seed, F.shape[0], (n_samples,), replace=False)]
        else:
            features_lo, features_hi = F.min(0), F.max(0)
            Q_rand = features_lo + (features_hi - features_lo) * jax.random.uniform(seed, (n_samples, d))
        nodes = jnp.concatenate([features_init[None], features_goal[None], Q_rand])
        N = nodes.shape[0]

        E_floor, E_ceiling = self._feature_energy_stats(tau_percentile)
        tau = (E_ceiling - E_floor) if tau is None else float(tau)
        cmax = E_floor + tau

        # Automatic neighbour detection following the original paper (Janson et al. 2015)
        if k is None:
            k = int(np.ceil(np.e * (1 + 1 / d) * np.log(N)))
        k = min(int(k), N - 1)

        # Compute self distance matrix of nodes by chunks to save memory and get the k 
        # closest neighbours indices
        near = self._knn_graph_fmt(nodes, k=k, chunk=chunk)

        # Compute edge cost
        cost_edge = self._edge_costs_fmt(nodes, near, E_floor, cmax, eps, quad=quad, chunk=chunk)

        # Get source, destination and weights for the undirected graph (each edge is bidirectional)
        src = jnp.repeat(jnp.arange(N, dtype=jnp.int32), k)  # src[i·k + j] = i
        dst = near.reshape(-1).astype(jnp.int32)  # dst[i·k + j] = near[i, j]
        src_e = jnp.concatenate([src, dst])
        dst_e = jnp.concatenate([dst, src])
        w_e = jnp.concatenate([cost_edge.reshape(-1), cost_edge.reshape(-1)])

        # Find shortest path from start to goal
        max_iters = N if max_iters is None else int(max_iters)
        cost_value, parent, iters = self._shortest_path_fmt(src_e, dst_e, w_e, n=N, max_iters=max_iters)
        goal_idx = 1
        if not np.isfinite(float(cost_value[goal_idx])):
            raise RuntimeError(f"goal unreachable with {N} nodes and k={k}; raise n_samples, k or tau")

        # Get the best path indices
        parent = np.asarray(parent)
        idx = [goal_idx]
        while idx[-1] != 0:
            idx.append(int(parent[idx[-1]]))
        idx = np.asarray(idx[::-1])
        path = np.asarray(nodes)[idx]

        # Compute energy profile along path and nodes in latent space
        energy, cum = self._path_profile(path, E_floor, quad)
        path_latent, particle_idx = self._path_to_latent(path, latent_init, latent_goal)

        info = dict(nodes=int(N), k=int(k), iters=int(iters), particle_idx=particle_idx, path_features=path,
                    energy_cum=cum, energy_floor=E_floor, tau=tau, graph_cost=float(cost_value[goal_idx]),
                    node_idx=idx)
        return path_latent, energy, cum, info
