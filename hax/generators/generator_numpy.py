import random
import numpy as np


class NumpyGenerator:
    def __init__(self, data):
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]

    def return_tf_dataset(self, preShuffle=False, shuffle=True, prefetch=-1, batch_size=8):
        import tensorflow_datasets as tfds
        import tensorflow as tf
        tf.config.set_visible_devices([], device_type='GPU')
        with tf.device("/CPU:0"):
            idx = np.arange(self.data.shape[0])
            if preShuffle:
                np.random.shuffle(idx)
            dataset = tf.data.Dataset.from_tensor_slices((self.data, idx))
            if shuffle:
                dataset = dataset.shuffle(len(idx))
            if prefetch == -1:
                prefetch = tf.data.AUTOTUNE
            return tfds.as_numpy(dataset.batch(batch_size).prefetch(prefetch))

    def return_grain_dataset(self, preShuffle=False, shuffle=True, batch_size=8, num_epochs=None, num_threads=1, num_workers=4):
        import grain

        idx = np.arange(self.data.shape[0])
        if preShuffle:
            np.random.shuffle(idx)
            data = self.data[idx]
        else:
            data = self.data

        class NumpyDataSource:
            def __init__(self, data):
                self._data = data

            def __len__(self):
                return len(self._data)

            def __getitem__(self, idx):
                return self._data[idx], idx

        source = NumpyDataSource(data)
        dataset = grain.MapDataset.source(source)

        # Shuffling type
        if shuffle:
            seed = random.randint(0, 2 ** 32 - 1)
            dataset = dataset.shuffle(seed=seed)

        dataset = dataset.repeat(num_epochs)

        read_options = grain.ReadOptions(num_threads=num_threads, prefetch_buffer_size=1)

        dataset = dataset.to_iter_dataset(read_options=read_options).batch(batch_size)

        if num_workers == -1:
            performance_config = grain.experimental.pick_performance_config(
                ds=dataset,
                ram_budget_mb=1024,
                max_workers=12,
                max_buffer_size=None
            )
            mp_options = performance_config.multiprocessing_options
        else:
            mp_options = grain.multiprocessing.MultiprocessingOptions(num_workers=num_workers, per_worker_buffer_size=2)

        dataset = dataset.mp_prefetch(options=mp_options)

        return dataset


class ArrayListGenerator:
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data[0].shape[0]

    def __getitem__(self, idx):
        return [np_array[idx] for np_array in self.data]

    def return_tf_dataset(self, preShuffle=False, shuffle=True, prefetch=-1, batch_size=8):
        import tensorflow_datasets as tfds
        import tensorflow as tf
        tf.config.set_visible_devices([], device_type='GPU')
        with tf.device("/CPU:0"):
            idx = np.arange(self.data[0].shape[0])
            if preShuffle:
                np.random.shuffle(idx)
            dataset = tf.data.Dataset.from_tensor_slices((tuple(self.data), idx))
            if shuffle:
                dataset = dataset.shuffle(len(idx))
            if prefetch == -1:
                prefetch = tf.data.AUTOTUNE
            return tfds.as_numpy(dataset.batch(batch_size).prefetch(prefetch))

    def return_grain_dataset(self, preShuffle=False, shuffle=True, batch_size=8, num_epochs=None, num_threads=1, num_workers=4):
        import grain

        idx = np.arange(self.data[0].shape[0])
        if preShuffle:
            np.random.shuffle(idx)
            data = [data[idx] for data in self.data]
        else:
            data = self.data

        class ListDataSource:
            def __init__(self, data):
                self._data = data

            def __len__(self):
                return self._data[0].shape[0]

            def __getitem__(self, idx):
                return [np_array[idx] for np_array in self._data], idx

        source = ListDataSource(data)
        dataset = grain.MapDataset.source(source)

        # Shuffling type
        if shuffle:
            seed = random.randint(0, 2 ** 32 - 1)
            dataset = dataset.shuffle(seed=seed)

        dataset = dataset.repeat(num_epochs)

        read_options = grain.ReadOptions(num_threads=num_threads, prefetch_buffer_size=1)

        dataset = dataset.to_iter_dataset(read_options=read_options).batch(batch_size)

        if num_workers == -1:
            performance_config = grain.experimental.pick_performance_config(
                ds=dataset,
                ram_budget_mb=1024,
                max_workers=None,
                max_buffer_size=None
            )
            mp_options = performance_config.multiprocessing_options
        else:
            mp_options = grain.multiprocessing.MultiprocessingOptions(num_workers=num_workers, per_worker_buffer_size=2)

        dataset = dataset.mp_prefetch(options=mp_options)

        return dataset