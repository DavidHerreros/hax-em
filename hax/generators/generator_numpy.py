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

    def return_tf_dataset(self, preShuffle=False, shuffle=True, prefetch=5, batch_size=8):
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
            return tfds.as_numpy(dataset.batch(batch_size).prefetch(prefetch))



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