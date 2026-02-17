import os
from etils import epath
import importlib
import random

import jax
from flax import nnx
import orbax.checkpoint as ocp
import cloudpickle

from hax.utils import bcolors


class NeuralNetworkCheckpointer:

    @classmethod
    def save(cls, model, checkpoint_path):
        os.makedirs(checkpoint_path, exist_ok=True)

        checkpoint_path = ocp.test_utils.erase_and_create_empty(os.path.abspath(os.path.join(checkpoint_path, 'checkpoints')))

        # Save model configuration
        with open(checkpoint_path / "config", "wb") as binary_file:
            cloudpickle.dump(model.config, binary_file)

        # Save model state
        _, state  = nnx.split(model)
        checkpointer = ocp.StandardCheckpointer()
        checkpointer.save(checkpoint_path / 'state', state)
        checkpointer.wait_until_finished()

    @classmethod
    def load(cls, checkpoint_path):
        checkpoint_path = epath.Path(os.path.abspath(os.path.join(checkpoint_path, 'checkpoints')))

        # Read config file
        with open(checkpoint_path / "config", "rb") as binary_file:
            config = cloudpickle.load(binary_file)

        # Instantiate model
        target = config.pop('_target_')
        module_path, class_name = target.rsplit('.', 1)
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)
        model = model_class(**config, rngs=nnx.Rngs(jax.random.PRNGKey(random.randint(0, 2 ** 32 - 1))))

        # Restore state
        checkpointer = ocp.StandardCheckpointer()
        _, state  = nnx.split(model)
        restored_state = checkpointer.restore(checkpoint_path / 'state', state)
        nnx.update(model, restored_state)

        model.eval()
        return model

    @classmethod
    def save_intermediate(cls, graphdef, state, checkpoint_path, epoch=None):
        os.makedirs(checkpoint_path, exist_ok=True)

        checkpoint_path = ocp.test_utils.erase_and_create_empty(os.path.abspath(os.path.join(checkpoint_path, 'checkpoints')))

        # Config file
        training_bundle = nnx.merge(graphdef, state)
        model = training_bundle[0]
        config = model.config

        # Save model configuration
        with open(checkpoint_path / "config", "wb") as binary_file:
            cloudpickle.dump((config, epoch), binary_file)

        # Save model state
        _, state = nnx.split(model)
        checkpointer = ocp.StandardCheckpointer()
        checkpointer.save(checkpoint_path / 'state', state)
        checkpointer.wait_until_finished()

    @classmethod
    def load_intermediate(cls, checkpoint_path, *optimizers: nnx.Optimizer, return_as_model=False):
        checkpoint_path = epath.Path(os.path.abspath(os.path.join(checkpoint_path, 'checkpoints')))

        with open(checkpoint_path / "config", "rb") as binary_file:
            config, epoch = cloudpickle.load(binary_file)

        # Instantiate model
        target = config.pop('_target_')
        module_path, class_name = target.rsplit('.', 1)
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)
        model = model_class(**config, rngs=nnx.Rngs(jax.random.PRNGKey(random.randint(0, 2 ** 32 - 1))))

        # Training bundle
        training_bundle = (model, *optimizers)

        # Restore state
        checkpointer = ocp.StandardCheckpointer()
        _, state = nnx.split(model)
        restored_state = checkpointer.restore(checkpoint_path / 'state', state)
        nnx.update(training_bundle, restored_state)

        if return_as_model:
            return *training_bundle, epoch
        else:
            return *nnx.split(model), epoch
