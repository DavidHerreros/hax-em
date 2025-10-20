import os
import sys

import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from jax import numpy as jnp
from jax.tree_util import tree_map
from xmipp_metadata.metadata import XmippMetaData
from hax.utils.loggers import bcolors


class MetaDataGenerator:
    def __init__(self, file, mode=None):
        self.file = file
        self.md = XmippMetaData(file)
        self.mode = mode

        # Generator mode
        if mode == "tomo" or self.md.isMetaDataLabel("subtomo_labels"):
            unique_labels = np.unique(self.md[:, "subtomo_labels"]).astype(int)
            self.sinusoid_tabled = get_sinusoid_encoding_table(np.amax(unique_labels), 100)
        else:
            self.sinusoid_tabled = np.zeros(len(self.md))

    def __len__(self):
        return len(self.md)

    def __getitem__(self, idx):
        if self.mode == "tomo":
            return self.md.getMetaDataImage(idx)[..., None], self.sinusoid_tabled[idx], idx
        else:
            return self.md.getMetaDataImage(idx)[..., None], idx

    def return_tf_dataset(self, preShuffle=False, shuffle=True, prefetch=-1, batch_size=8, mmap=True, mmap_output_dir=None):
        import tensorflow_datasets as tfds
        import tensorflow as tf
        tf.config.set_visible_devices([], device_type='GPU')
        with tf.device("/CPU:0"):
            file_idx = np.arange(len(self.md))
            if preShuffle:
                np.random.shuffle(file_idx)
            if mmap:
                drop_reminder = True
                from mmap_ninja import numpy as np_ninja

                if mmap_output_dir is None:
                    mmap_output_dir = os.path.join(os.path.dirname(self.file), "images_mmap")
                else:
                    mmap_output_dir = os.path.join(mmap_output_dir, "images_mmap")

                if not os.path.isdir(mmap_output_dir):
                    print(f"{bcolors.OKCYAN}\n###### Creating MMAP from images... ######")
                    np_ninja.from_generator(
                        out_dir=mmap_output_dir,
                        sample_generator=map(self.md.getMetaDataImage, file_idx),
                        batch_size=1024,
                        verbose=True
                    )
                images = np_ninja.open_existing(mmap_output_dir)

                def _load_image(idx):
                    idx = idx.astype(np.int64)
                    image = images[idx][..., None]
                    if self.mode == "tomo":
                        subtomo_label = self.sinusoid_table[self.md[idx, "subtomo_labels"].astype(int) - 1]
                        return (image, subtomo_label), (idx, idx)
                    else:
                        return image, idx

                def map_fn(i):
                    if self.mode == "tomo":
                        (image, subtomo_labels), (idx, idl) = tf.numpy_function(_load_image, [i], (tf.float32, tf.int64, tf.int64, tf.int64))
                        image.set_shape((batch_size,) + images[0][..., None].shape)
                        idx.set_shape([batch_size,])
                        subtomo_labels.set_shape([batch_size,])
                        idl.set_shape([batch_size,])
                        return (image, subtomo_labels), (idx, idx)
                    else:
                        image, idx = tf.numpy_function(_load_image, [i], (tf.float32, tf.int64))
                        image.set_shape((batch_size,) + images[0][..., None].shape)
                        idx.set_shape([batch_size,])
                        return image, idx

                dataset = tf.data.Dataset.from_tensor_slices(file_idx)
            else:
                drop_reminder = False

                def build_ram_slab(N, H, W, batch_size=4096, threads=16):
                    X = np.empty((N, H, W), dtype=np.float32)

                    def _ranges():
                        for s in range(0, N, batch_size):
                            e = min(s + batch_size, N)
                            yield s, e, np.arange(s, e, dtype=np.int64)

                    with ThreadPoolExecutor(max_workers=threads) as ex:
                        futs = []
                        for s, e, idx in _ranges():
                            futs.append((s, e, ex.submit(self.md.getMetaDataImage, idx)))
                        with tqdm(total=N, file=sys.stdout, ascii=" >=", colour="green") as pbar:
                            for s, e, fut in futs:
                                X[s:e] = fut.result()
                                pbar.update(e - s)
                    return X

                print(f"{bcolors.OKCYAN}\n###### Loading images to RAM... ######")
                H, W = self.md.getMetaDataImage(0).shape
                images = build_ram_slab(len(self.md), H, W, batch_size=4096, threads=8)[..., None]
                images = images[file_idx]  # Reorder according to preShuffle
                if self.mode == "tomo":
                    subtomo_labels = self.sinusoid_table[self.md[file_idx, "subtomo_labels"].astype(int) - 1]
                    dataset = tf.data.Dataset.from_tensor_slices(((images, subtomo_labels), (file_idx, file_idx)))
                else:
                    dataset = tf.data.Dataset.from_tensor_slices((images, file_idx))
            if shuffle:
                dataset = dataset.shuffle(len(file_idx))
            if prefetch == -1:
                prefetch = tf.data.AUTOTUNE
            dataset = dataset.batch(batch_size, drop_remainder=drop_reminder)
            if mmap:
                dataset = dataset.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
            return tfds.as_numpy(dataset.prefetch(prefetch))

    def return_torch_dataset(self, shuffle=True, batch_size=8):
        from torch.utils.data import default_collate, DataLoader

        def numpy_collate(batch):
            return tree_map(np.asarray, default_collate(batch))

        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=0, collate_fn=numpy_collate)

    def return_grain_dataset(self, preShuffle=False, shuffle=True, batch_size=8):
        import grain

        class CustomSource(grain.sources.RandomAccessDataSource):
            def __init__(self, data, labels, subtomo_labels=None):
                self.data = data
                self.labels = labels
                self.subtomo_labels = subtomo_labels

            def __getitem__(self, idx):
                if self.subtomo_labels is None:
                    return self.data[idx], self.labels[idx]
                else:
                    return (self.data[idx], self.subtomo_labels[idx]), self.labels[idx]

            def __len__(self):
                return len(self.data)

        file_idx = np.arange(len(self.md))
        if preShuffle:
            np.random.shuffle(file_idx)
        images = self.md.getMetaDataImage(file_idx)[..., None]
        if self.mode == "tomo":
            subtomo_labels = self.sinusoid_table[self.md[file_idx, "subtomo_labels"].astype(int) - 1]
            dataset = grain.MapDataset.source(CustomSource(images, file_idx, subtomo_labels))
        else:
            dataset = grain.MapDataset.source(CustomSource(images, file_idx))
        if shuffle:
            seed = random.randint(0, 2 ** 32 - 1)
            dataset = dataset.shuffle(seed=seed)
        return dataset.to_iter_dataset().batch(batch_size)

def extract_columns(md, hasCTF=None):
    hasCTF = md.isMetaDataLabel("ctfDefocusU") if hasCTF is None else hasCTF

    columns = {}
    columns["euler_angles"] = jnp.array(md.getMetaDataColumns(["angleRot", "angleTilt", "anglePsi"]).astype(jnp.float32))
    columns["shifts"] = jnp.array(md.getMetaDataColumns(["shiftX", "shiftY"]).astype(jnp.float32))
    if hasCTF:
        columns["ctfDefocusU"] = jnp.array(md.getMetaDataColumns("ctfDefocusU").astype(jnp.float32))
        columns["ctfDefocusV"] = jnp.array(md.getMetaDataColumns("ctfDefocusV").astype(jnp.float32))
        columns["ctfDefocusAngle"] = jnp.array(md.getMetaDataColumns("ctfDefocusAngle").astype(jnp.float32))
        columns["ctfSphericalAberration"] = jnp.array(md.getMetaDataColumns("ctfSphericalAberration").astype(jnp.float32))
        columns["ctfVoltage"] = jnp.array(md.getMetaDataColumns("ctfVoltage").astype(jnp.float32))
    return columns


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """
    numpy sinusoid position encoding of Transformer model.
    params:
        n_position(n):number of positions
        d_hid(m): dimension of embedding vector
        padding_idx:set 0 dimension
    return:
        sinusoid_table(n*m):numpy array
    """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return sinusoid_table
