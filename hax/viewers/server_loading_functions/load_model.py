import numpy as np
import types
import jax

from xmipp_metadata.image_handler import ImageHandler

from hax.checkpointer import NeuralNetworkCheckpointer


class HeterogeneityProgramInterface:
    def __init__(self, _path_template: str, _program_loading_params: dict):
        self.model = self.prepare_heterogeneity_program(**_program_loading_params)
        self.path_template = _path_template

    def prepare_heterogeneity_program(self, **kwargs) -> object:
        model = NeuralNetworkCheckpointer.load(kwargs.pop("pickled_nn"))
        return model

    def decode_state_from_latent(self, latent: np.array) -> None:
        import numpy as np

        if latent.ndim == 1:
            latent = latent[None, ...]

        idx = 1
        for latent_vector in latent:
            state = np.array(self.model.decode_volume(latent_vector))
            ImageHandler().write(state, filename=self.path_template.format(idx), overwrite=True)
            idx += 1
