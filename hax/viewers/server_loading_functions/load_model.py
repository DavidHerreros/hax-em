import numpy as np

from xmipp_metadata.image_handler import ImageHandler

from hax.checkpointer import NeuralNetworkCheckpointer


class HeterogeneityProgramInterface:
    def __init__(self, _path_template: str, _program_loading_params: dict):
        self.model = self.prepare_heterogeneity_program(**_program_loading_params)
        self.path_template = _path_template

    def prepare_heterogeneity_program(self, **kwargs) -> object:
        # Load neural network (note it MUST be saved in pickle mode to make this script general)
        model = NeuralNetworkCheckpointer.load(None, kwargs.pop("pickled_nn"), mode="pickle")

        return model

    def decode_state_from_latent(self, latent: np.array) -> None:
        import numpy as np

        states = np.array(self.model.decode_states(latent))

        idx = 1
        for state in states:
            ImageHandler().write(state, filename=self.path_template.format(idx), overwrite=True)
            idx += 1
