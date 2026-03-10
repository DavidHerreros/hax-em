import os
import sys
import struct
from glob import glob

import numpy as np
import random
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from multiprocessing import get_context
from tqdm import tqdm
from jax import numpy as jnp
from jax.tree_util import tree_map
from xmipp_metadata.metadata import XmippMetaData
from hax.utils.loggers import bcolors


def _write_one_shard_array_record(path, image_indices, getImage_fn, dtype=np.float16):
    """
    Worker: loads images for a shard and writes a single ArrayRecord file.
    """
    from array_record.python.array_record_module import ArrayRecordWriter

    # Load images for this shard
    imgs = getImage_fn(image_indices)
    imgs = np.ascontiguousarray(imgs.astype(dtype, copy=False))

    # Precompute dtype bytes once per shard
    dtype_str = imgs.dtype.str.encode("utf-8")
    label_struct = struct.Struct("<I")  # unsigned int label prefix

    # If all images share shape (typical), precompute header too
    first_shape = imgs[0].shape
    same_shape = True
    for k in range(1, len(imgs)):
        if imgs[k].shape != first_shape:
            same_shape = False
            break

    if same_shape:
        shape = first_shape
        header = struct.pack(
            f"<I{len(dtype_str)}sI{len(shape)}I",
            len(dtype_str), dtype_str, len(shape), *shape
        )

    writer = ArrayRecordWriter(path, "group_size:1,uncompressed")
    try:
        for img, label in zip(imgs, image_indices):
            raw = img.tobytes()

            if not same_shape:
                shape = img.shape
                header = struct.pack(
                    f"<I{len(dtype_str)}sI{len(shape)}I",
                    len(dtype_str), dtype_str, len(shape), *shape
                )

            record_bytes = label_struct.pack(int(label)) + header + raw
            writer.write(record_bytes)
    finally:
        writer.close()

def parse_and_decompress(record_bytes):
    label = struct.unpack('<I', record_bytes[:4])[0]
    cursor = 4
    dtype_len = struct.unpack('<I', record_bytes[cursor:cursor + 4])[0]
    cursor += 4
    dtype_str = record_bytes[cursor:cursor + dtype_len]
    cursor += dtype_len
    ndim = struct.unpack('<I', record_bytes[cursor:cursor + 4])[0]
    cursor += 4
    shape_format = f'<{ndim}I'
    shape_size = struct.calcsize(shape_format)
    shape = struct.unpack(shape_format, record_bytes[cursor:cursor + shape_size])
    cursor += shape_size
    data_bytes = record_bytes[cursor:]
    image = np.frombuffer(data_bytes, dtype=np.dtype(dtype_str)).reshape(shape)
    return image.astype(np.float32, copy=False)[..., None], label


def _write_one_shard_mmap(path, image_indices, getImage_fn, dtype=np.float16):
    from mmap_ninja import numpy as np_ninja

    def getImage_dtype():
        for idx in image_indices:
            yield getImage_fn(idx).astype(dtype)

    np_ninja.from_generator(
        out_dir=path,
        sample_generator=getImage_dtype(),
        batch_size=2048,
        verbose=False
    )


class MetaDataGenerator:
    def __init__(self, file, mode=None):
        self.file = file
        self.md = XmippMetaData(file)
        self.mode = mode

        # Generator mode
        if mode == "tomo" or self.md.isMetaDataLabel("subtomo_labels"):
            self.mode = "tomo"
            unique_labels = np.unique(self.md[:, "subtomo_labels"]).astype(int)
            self.sinusoid_table = get_sinusoid_encoding_table(np.amax(unique_labels), 100)
        else:
            self.sinusoid_table = np.zeros(len(self.md), dtype=np.float32)

    def __len__(self):
        return len(self.md)

    def __getitem__(self, idx):
        if self.mode == "tomo":
            return self.md.getMetaDataImage(idx)[..., None], self.sinusoid_tabled[idx], idx
        else:
            return self.md.getMetaDataImage(idx)[..., None], idx

    def load_images_to_ram(self, images_order=None):
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

        if images_order is None:
            images_order = np.arange(len(self.md))

        print(f"{bcolors.OKCYAN}\n###### Loading images to RAM... ######")
        H, W = self.md.getMetaDataImage(0).shape
        images = build_ram_slab(len(self.md), H, W, batch_size=4096, threads=8)[..., None]
        return images[images_order]

    def load_images_to_array_record(self, mmap_output_dir=None, images_order=None, multiple_files=True, shard_size=10000,
                                    num_workers=None, precision=np.float16, group_size=1, batch_reading_size=20000):
        if images_order is None:
            images_order = np.arange(len(self.md))

        # If we already have shards, just return the datasource
        existing = glob(os.path.join(mmap_output_dir, "dataset-*.arrayrecord"))
        if not existing:
            print(f"{bcolors.OKCYAN}\n###### Creating Array Record from images... ######")

            if not multiple_files:
                # Single file.
                shard_name = "dataset-00000.arrayrecord"
                path = os.path.join(mmap_output_dir, shard_name)
                from array_record.python.array_record_module import ArrayRecordWriter
                import struct

                writer = ArrayRecordWriter(path, f"group_size:{group_size},uncompressed")
                try:
                    label_struct = struct.Struct("<I")
                    with tqdm(total=len(images_order), file=sys.stdout, ascii=" >=", colour="green") as pbar:
                        for i in range(0, len(images_order), batch_reading_size):
                            idxs = images_order[i:i + batch_reading_size]
                            imgs = self.md.getMetaDataImage(idxs)
                            imgs = np.ascontiguousarray(imgs.astype(precision, copy=False))

                            dtype_str = imgs.dtype.str.encode("utf-8")
                            # assume same shape in batch; recompute header per batch
                            shape = imgs[0].shape
                            header = struct.pack(
                                f"<I{len(dtype_str)}sI{len(shape)}I",
                                len(dtype_str), dtype_str, len(shape), *shape
                            )

                            for img, label in zip(imgs, idxs):
                                record_bytes = label_struct.pack(int(label)) + header + img.tobytes()
                                writer.write(record_bytes)
                                pbar.update(1)
                finally:
                    writer.close()

            else:
                # Multiple files: write shards in parallel
                if num_workers is None:
                    num_workers = os.cpu_count() or 4

                # Build shard jobs
                shards = []
                n = len(images_order)
                shard_idx = 0
                for start in range(0, n, shard_size):
                    end = min(start + shard_size, n)
                    idxs = np.asarray(images_order[start:end])
                    shard_name = f"dataset-{shard_idx:05d}.arrayrecord"
                    path = os.path.join(mmap_output_dir, shard_name)
                    shards.append((path, idxs))
                    shard_idx += 1

                # Submit processes
                with tqdm(total=n, file=sys.stdout, ascii=" >=", colour="green") as pbar:
                    with ProcessPoolExecutor(max_workers=num_workers, mp_context=get_context('forkserver')) as ex:
                        futures = [
                            ex.submit(_write_one_shard_array_record, path, idxs, self.md.getMetaDataImage)
                            for (path, idxs) in shards
                        ]

                        # Update progress by shard completion (exact per-record progress would require IPC)
                        for fut, (_, idxs) in zip(as_completed(futures), shards):
                            fut.result()  # raise if any error
                            pbar.update(len(idxs))

    def load_images_to_mmap(self, mmap_output_dir=None, images_order=None, shard_size=10000, num_workers=None,
                            precision=np.float16, multiple_files=True):
        if multiple_files:

            # If we already have shards, just return the datasource
            existing = glob(os.path.join(mmap_output_dir, "dataset-*"))

            if not existing:
                print(f"{bcolors.OKCYAN}\n###### Creating MMAP from images... ######")

                # Multiple files: write shards in parallel
                if num_workers is None:
                    num_workers = os.cpu_count() or 4

                # Build shard jobs
                shards = []
                n = len(images_order)
                shard_idx = 0
                for start in range(0, n, shard_size):
                    end = min(start + shard_size, n)
                    idxs = np.asarray(images_order[start:end])
                    shard_name = f"dataset-{shard_idx:05d}"
                    path = os.path.join(mmap_output_dir, shard_name)
                    shards.append((path, idxs))
                    shard_idx += 1

                # Submit processes
                with tqdm(total=n, file=sys.stdout, ascii=" >=", colour="green") as pbar:
                    with ProcessPoolExecutor(max_workers=num_workers) as ex:
                        futures = [
                            ex.submit(_write_one_shard_mmap, path, idxs, self.md.getMetaDataImage, precision)
                            for (path, idxs) in shards
                        ]

                        # Update progress by shard completion (exact per-record progress would require IPC)
                        for fut, (_, idxs) in zip(as_completed(futures), shards):
                            fut.result()  # raise if any error
                            pbar.update(len(idxs))

            else:
                if not os.path.isdir(mmap_output_dir):
                    from mmap_ninja import numpy as np_ninja
                    np_ninja.from_generator(
                        out_dir=mmap_output_dir,
                        sample_generator=map(self.md.getMetaDataImage, images_order),
                        batch_size=4096,
                        verbose=True
                    )

    def return_tf_dataset(self, preShuffle=False, shuffle=True, prefetch=-1, batch_size=8, mmap=True, mmap_output_dir=None, split_fraction=None):
        import tensorflow_datasets as tfds
        import tensorflow as tf

        tf.config.set_visible_devices([], device_type='GPU')
        with tf.device("/CPU:0"):
            file_idx = np.arange(len(self.md))

            if preShuffle:
                np.random.shuffle(file_idx)

            if mmap:
                from mmap_ninja import numpy as np_ninja

                self.load_images_to_mmap(mmap_output_dir=mmap_output_dir, images_order=None)
                images = np_ninja.open_existing(mmap_output_dir)

                def _load_image(idx):
                    idx = idx.astype(np.int64)
                    image = images[idx][..., None]
                    if self.mode == "tomo":
                        subtomo_label = self.sinusoid_table[self.md[idx, "subtomo_labels"].astype(int) - 1]
                        return image, subtomo_label, idx
                    else:
                        return image, idx

                def map_fn(i):
                    if self.mode == "tomo":
                        (image, subtomo_labels, idx) = tf.numpy_function(_load_image, [i], (tf.float32, tf.float32, tf.int64))
                        image.set_shape((None,) + images[0][..., None].shape)
                        idx.set_shape([None,])
                        subtomo_labels.set_shape([None, 100])
                        return (image, subtomo_labels), idx
                    else:
                        image, idx = tf.numpy_function(_load_image, [i], (tf.float32, tf.int64))
                        image.set_shape((None,) + images[0][..., None].shape)
                        idx.set_shape([None,])
                        return image, idx

                dataset = tf.data.Dataset.from_tensor_slices(file_idx)
            else:
                images = self.load_images_to_ram(images_order=file_idx)

                if self.mode == "tomo":
                    subtomo_labels = self.sinusoid_table[self.md[file_idx, "subtomo_labels"].astype(int) - 1]
                    dataset = tf.data.Dataset.from_tensor_slices(((images, subtomo_labels), file_idx))
                else:
                    dataset = tf.data.Dataset.from_tensor_slices((images, file_idx))

            if prefetch == -1:
                prefetch = tf.data.AUTOTUNE

            if split_fraction is None:
                if shuffle:
                    dataset = dataset.shuffle(len(file_idx))

                dataset = dataset.batch(batch_size)

                if mmap:
                    dataset = dataset.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)

                return tfds.as_numpy(dataset.prefetch(prefetch))
            else:
                split_at = int(split_fraction[0] * len(file_idx))
                dataset_train = dataset.take(split_at)
                dataset_validation = dataset.skip(split_at)

                if shuffle:
                    dataset = dataset.shuffle(len(file_idx))
                    dataset_train = dataset_train.shuffle(split_at)

                dataset = dataset.batch(batch_size)
                dataset_train = dataset_train.batch(batch_size)
                dataset_validation = dataset_validation.batch(batch_size)

                if mmap:
                    dataset = dataset.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
                    dataset_train = dataset_train.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
                    dataset_validation = dataset_validation.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)

                return tfds.as_numpy(dataset.prefetch(prefetch)), tfds.as_numpy(dataset_train.prefetch(prefetch)), tfds.as_numpy(dataset_validation.prefetch(prefetch))


    def return_torch_dataset(self, shuffle=True, preShuffle=True, batch_size=8, mmap=True, mmap_output_dir=None):
        import torch
        from torch.utils.data import default_collate, DataLoader, TensorDataset

        file_idx = np.arange(len(self.md))

        if preShuffle:
            np.random.shuffle(file_idx)

        if mmap:
            images = self.load_images_to_mmap(mmap_output_dir=mmap_output_dir, images_order=file_idx)
        else:
            images = self.load_images_to_ram(images_order=file_idx)

        if self.mode == "tomo":
            subtomo_labels = self.sinusoid_table[self.md[file_idx, "subtomo_labels"].astype(int) - 1]
            dataset = TensorDataset(torch.from_numpy(images), torch.from_numpy(subtomo_labels), torch.from_numpy(file_idx))
        else:
            dataset = TensorDataset(torch.from_numpy(images), torch.from_numpy(file_idx))

        def numpy_collate(batch):
            return tree_map(np.asarray, default_collate(batch))

        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=16, collate_fn=numpy_collate, persistent_workers=True,
                          pin_memory=True)

    def prepare_grain_array_record(self, mmap_output_dir=None, preShuffle=False, num_workers=16, precision=np.float16, group_size=1,
                                   shard_size=10000):
        self.grain_dataset_type = "ArrayRecord"

        # Prepare folder to save data
        if mmap_output_dir is None:
            self.mmap_output_dir = os.path.join(os.path.dirname(self.file), "images_mmap_grain")
        else:
            self.mmap_output_dir = os.path.join(mmap_output_dir, "images_mmap_grain")
        os.makedirs(self.mmap_output_dir, exist_ok=True)

        # Pre shuffle data before writing
        file_idx = np.arange(len(self.md))
        if preShuffle:
            np.random.shuffle(file_idx)

        self.load_images_to_array_record(mmap_output_dir=self.mmap_output_dir, images_order=file_idx, num_workers=num_workers,
                                         precision=precision, group_size=group_size, shard_size=shard_size)

    def prepare_grain_mmap(self, mmap_output_dir=None, preShuffle=False, num_workers=16, shard_size=10000, precision=np.float16,
                           multiple_files=True):
        self.grain_dataset_type = "MMAP"

        # Prepare folder to save data
        if mmap_output_dir is None:
            self.mmap_output_dir = os.path.join(os.path.dirname(self.file), "images_mmap")
        else:
            self.mmap_output_dir = os.path.join(mmap_output_dir, "images_mmap")
        os.makedirs(self.mmap_output_dir, exist_ok=True)

        # Pre shuffle data before writing
        images_order = np.arange(len(self.md))
        if preShuffle:
            np.random.shuffle(images_order)

        self.load_images_to_mmap(mmap_output_dir=self.mmap_output_dir, images_order=images_order, num_workers=num_workers,
                                 shard_size=shard_size, precision=precision, multiple_files=multiple_files)

    def return_grain_dataset(self, shuffle="global", batch_size=8, num_epochs=1, num_threads=1, num_workers=16,
                             split_fraction=None, load_to_ram=False):
        import grain
        from array_record.python.array_record_data_source import ArrayRecordDataSource

        self.grain_dataset_type = "RAM" if load_to_ram else self.grain_dataset_type

        # Get sources
        if self.grain_dataset_type == "ArrayRecord":
            shard_files = glob(os.path.join(self.mmap_output_dir, "dataset-*.arrayrecord"))
            shard_files.sort()

            if split_fraction is not None:
                split_point = int(split_fraction[0] * len(shard_files))
                sources_train = ArrayRecordDataSource(shard_files[:split_point], reader_options={"index_storage_option": "in_memory"})
                sources_val = ArrayRecordDataSource(shard_files[split_point:], reader_options={"index_storage_option": "in_memory"})
                dataset_train = grain.MapDataset.source(sources_train)
                dataset_val = grain.MapDataset.source(sources_val)
            else:
                sources_train = ArrayRecordDataSource(shard_files, reader_options={"index_storage_option": "in_memory"})
                dataset_train = grain.MapDataset.source(sources_train)

        elif self.grain_dataset_type == "MMAP":
            from mmap_ninja import numpy as np_ninja

            # Class to handle mmap_ninja shards
            class LazyNinjaGrainSource:
                def __init__(self, shard_paths):
                    self.shard_paths = shard_paths

                    self.shard_lengths = []
                    for p in shard_paths:
                        mmap = np_ninja.open_existing(p, mode="r")
                        self.shard_lengths.append(len(mmap))

                    self.total_len = sum(self.shard_lengths)
                    self.cumulative_indices = np.cumsum(self.shard_lengths)

                def _get_shard(self, shard_idx):
                    return np_ninja.open_existing(self.shard_paths[shard_idx], mode="r")

                def __len__(self):
                    return self.total_len

                def __getitem__(self, idx):
                    if idx < 0 or idx >= self.total_len:
                        raise IndexError

                    # Find the shard index
                    shard_idx = np.searchsorted(self.cumulative_indices, idx, side='right')

                    # Calculate local index
                    if shard_idx == 0:
                        local_idx = idx
                    else:
                        local_idx = idx - self.cumulative_indices[shard_idx - 1]

                    # Get the shard lazily (using the LRU cache)
                    shard = self._get_shard(shard_idx)

                    # Access data (returns a numpy array)
                    return shard[local_idx][..., None].astype(np.float32), idx

            shard_paths = glob(os.path.join(self.mmap_output_dir, "dataset-*"))
            shard_paths.sort()

            if split_fraction is not None:
                split_point = int(split_fraction[0] * len(shard_paths))
                sources_train = LazyNinjaGrainSource(shard_paths[split_point:])
                sources_val = LazyNinjaGrainSource(shard_paths[:split_point])
                dataset_train = grain.MapDataset.source(sources_train)
                dataset_val = grain.MapDataset.source(sources_val)
            else:
                sources_train = LazyNinjaGrainSource(shard_paths)
                dataset_train = grain.MapDataset.source(sources_train)

        elif self.grain_dataset_type == "RAM":

            images = self.load_images_to_ram(np.arange(len(self.md)))

            class NumpyDataSource:
                def __init__(self, data):
                    self._data = data

                def __len__(self):
                    return len(self._data)

                def __getitem__(self, idx):
                    return self._data[idx], idx

            if split_fraction is not None:
                split_point = int(split_fraction[0] * len(images))
                sources_train = NumpyDataSource(images[split_point:])
                sources_val = NumpyDataSource(images[:split_point])
                dataset_train = grain.MapDataset.source(sources_train)
                dataset_val = grain.MapDataset.source(sources_val)
            else:
                sources_train = NumpyDataSource(images)
                dataset_train = grain.MapDataset.source(sources_train)

        else:
            raise ValueError("Unknown grain dataset type")

        if split_fraction is not None:
            split_point = int(split_fraction[0] * len(self.md))
            dataset_train = dataset_train[:split_point]
            dataset_val = dataset_val[split_point:]

        # Shuffling type
        if shuffle == "global":
            seed = random.randint(0, 2 ** 32 - 1)
            dataset_train = dataset_train.shuffle(seed=seed)
            if split_fraction is not None:
                dataset_val = dataset_val.shuffle(seed=seed)

            if self.grain_dataset_type == "ArrayRecord":
                dataset_train = dataset_train.map(parse_and_decompress)
                if split_fraction is not None:
                    dataset_val = dataset_val.map(parse_and_decompress)

            dataset_train = dataset_train.repeat(num_epochs)
            if split_fraction is not None:
                dataset_val = dataset_val.repeat(num_epochs)

            if num_threads > 1:
                read_options = grain.ReadOptions(num_threads=num_threads, prefetch_buffer_size=1)
            else:
                read_options = None


            dataset_train = dataset_train.to_iter_dataset(read_options=read_options).batch(batch_size)
            if split_fraction is not None:
                dataset_val = dataset_val.to_iter_dataset(read_options=read_options).batch(batch_size)

            if num_workers == -1:
                performance_config = grain.experimental.pick_performance_config(
                    ds=dataset_train,
                    ram_budget_mb=1024,
                    max_workers=12,
                    max_buffer_size=None
                )
                mp_options = performance_config.multiprocessing_options
            else:
                mp_options = grain.multiprocessing.MultiprocessingOptions(num_workers=num_workers, per_worker_buffer_size=2)
            dataset_train = dataset_train.mp_prefetch(options=mp_options)
            if split_fraction is not None:
                dataset_val = dataset_val.mp_prefetch(options=mp_options)

        elif shuffle == "global_data_loader":
            # Operations
            operations = []
            if self.grain_dataset_type == "ArrayRecord":
                class ParseAndDecompress(grain.transforms.Map):
                    def map(self, x):
                        return parse_and_decompress(x)
                operations = [ParseAndDecompress()]
            operations.append(grain.transforms.Batch(batch_size=batch_size))

            # Index sampler
            sampler_train = grain.samplers.IndexSampler(
                num_records=len(sources_train),
                shuffle=True,
                seed=random.randint(0, 2 ** 32 - 1),
                num_epochs=num_epochs,
                shard_options=grain.sharding.NoSharding()
            )
            if split_fraction is not None:
                sampler_val = grain.samplers.IndexSampler(
                    num_records=len(sources_val),
                    shuffle=True,
                    seed=random.randint(0, 2 ** 32 - 1),
                    num_epochs=1,
                    shard_options=grain.sharding.NoSharding()
                )

            # Workers detection
            if num_workers == -1:
                dataset_train = dataset_train.to_iter_dataset().batch(batch_size)
                performance_config = grain.experimental.pick_performance_config(
                    ds=dataset_train,
                    ram_budget_mb=1024,
                    max_workers=12,
                    max_buffer_size=None
                )
                num_workers = performance_config.multiprocessing_options.num_workers


            # Data loader
            # The DataLoader coordinates the workers and the sampler
            dataset_train = grain.DataLoader(
                data_source=sources_train,
                sampler=sampler_train,
                operations=operations,
                worker_count=num_workers,
            )
            if split_fraction is not None:
                dataset_val = grain.DataLoader(
                    data_source=sources_val,
                    sampler=sampler_val,
                    operations=operations,
                    worker_count=num_workers,
                )

        elif shuffle == "hierarchical":
            seed = random.randint(0, 2 ** 32 - 1)

            dataset_train = grain.experimental.WindowShuffleMapDataset(dataset_train, window_size=2048, seed=seed) 
            if split_fraction is not None:
                dataset_val = grain.experimental.WindowShuffleMapDataset(dataset_val, window_size=2048, seed=seed) 

            if self.grain_dataset_type == "ArrayRecord":
                dataset_train = dataset_train.map(parse_and_decompress)
                if split_fraction is not None:
                    dataset_val = dataset_val.map(parse_and_decompress)

            dataset_train = dataset_train.repeat(num_epochs)
            if split_fraction is not None:
                dataset_val = dataset_val.repeat(num_epochs)

            # dataset = dataset.to_iter_dataset()

            dataset_train = grain.experimental.InterleaveIterDataset(dataset_train, cycle_length=10)
            if split_fraction is not None:
                dataset_val = grain.experimental.InterleaveIterDataset(dataset_val, cycle_length=10)

            dataset_train = dataset_train.batch(batch_size)
            if split_fraction is not None:
                dataset_val = dataset_val.batch(batch_size)

            mp_options = grain.multiprocessing.MultiprocessingOptions(num_workers=16, per_worker_buffer_size=2)
            dataset_train = dataset_train.mp_prefetch(options=mp_options)
            if split_fraction is not None:
                dataset_val = dataset_val.mp_prefetch(options=mp_options)

        else:
            # Operations
            operations = []
            if self.grain_dataset_type == "ArrayRecord":
                class ParseAndDecompress(grain.transforms.Map):
                    def map(self, x):
                        return parse_and_decompress(x)
                operations = [ParseAndDecompress()]
            operations.append(grain.transforms.Batch(batch_size=batch_size))

            # Index sampler
            if num_epochs == 1:
                sampler_train = grain.samplers.SequentialSampler(
                    num_records=len(sources_train),
                    shard_options=grain.sharding.NoSharding()
                )
            else:
                sampler_train = grain.samplers.IndexSampler(
                    num_records=len(sources_train),
                    shuffle=False,
                    num_epochs=num_epochs,
                    shard_options=grain.sharding.NoSharding()
                )

            # Workers detection
            if num_workers == -1:
                dataset_train = dataset_train.to_iter_dataset().batch(batch_size)
                performance_config = grain.experimental.pick_performance_config(
                    ds=dataset_train,
                    ram_budget_mb=1024,
                    max_workers=12,
                    max_buffer_size=None
                )
                num_workers = performance_config.multiprocessing_options.num_workers


            # Data loader
            # The DataLoader coordinates the workers and the sampler
            dataset_train = grain.DataLoader(
                data_source=sources_train,
                sampler=sampler_train,
                operations=operations,
                worker_count=num_workers,
            )
            if split_fraction is not None:
                dataset_val = None  # Not implemented, no shuffling is intended to be used for prediction steps

        if split_fraction is not None:
            return dataset_train, dataset_val
        else:
            return dataset_train


def extract_columns(md, hasCTF=None, isTomo=None):
    hasCTF = md.isMetaDataLabel("ctfDefocusU") if hasCTF is None else hasCTF
    isTomo = md.isMetaDataLabel("subtomo_labels") if isTomo is None else isTomo

    columns = {}
    columns["euler_angles"] = jnp.array(md.getMetaDataColumns(["angleRot", "angleTilt", "anglePsi"]).astype(jnp.float32))
    columns["shifts"] = jnp.array(md.getMetaDataColumns(["shiftX", "shiftY"]).astype(jnp.float32))
    if hasCTF:
        columns["ctfDefocusU"] = jnp.array(md.getMetaDataColumns("ctfDefocusU").astype(jnp.float32))
        columns["ctfDefocusV"] = jnp.array(md.getMetaDataColumns("ctfDefocusV").astype(jnp.float32))
        columns["ctfDefocusAngle"] = jnp.array(md.getMetaDataColumns("ctfDefocusAngle").astype(jnp.float32))
        columns["ctfSphericalAberration"] = jnp.array(md.getMetaDataColumns("ctfSphericalAberration").astype(jnp.float32))
        columns["ctfVoltage"] = jnp.array(md.getMetaDataColumns("ctfVoltage").astype(jnp.float32))
    if isTomo:
        columns["subtomo_labels"] = jnp.array(md.getMetaDataColumns("subtomo_labels").astype(jnp.float32))
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

    return sinusoid_table.astype(np.float32)
