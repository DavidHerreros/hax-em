###### NOTE ######
#
#  This file/template can be copied to any desired location to simplify the connection of the annotate_space
#  viewer to your software.
#
#  You may pass the modified file to the viewer through the command line parameter
#  server_functions_path - keep in mind that you may need to pass as well the environment where your software is
#  installed through the parameter env_name. In this way, the viewer will automatically instantiate this class within
#  your environment, ensuring that your dependencies are available.
#
#  Remember to NOT import this file in your package, as it is not needed by the viewer. Annotate space will be able to
#  discover your class thanks to the provided server_functions_path parameter containing the path to the modified version
#  of this file
#
#  An example on how to implement this class can be found at hax/viewers/sever_loading_functions/load_model.py
#
#################


from abc import abstractmethod
import numpy as np


class HeterogeneityProgramInterface:
    def __init__(self, _path_template: str, _program_loading_params: dict):
        # _path_template is a string pointing to the path where the generated volume will be saved
        # _program_loading_params will contain any additional/custom parameters passed to annotate_space program through
        # the command line. The entries of the dictionary will keep the exact same name given to the parameters, which
        # are expected to be needed to load a method able to generate volumes from latent vectors. In the method
        # prepare_heterogeneity_program, you may access these parameters with the syntax kwargs.pop("myParameter")
        self.model = self.prepare_heterogeneity_program(**_program_loading_params)
        self.path_template = _path_template

    @abstractmethod
    def prepare_heterogeneity_program(self, **kwargs) -> object:
        # This method will be used to define the logic needed to load a program responsible for generating volumes from
        # latent vectors
        # The returned object will correspond to the loaded model
        pass

    @abstractmethod
    def decode_state_from_latent(self, latent: np.array) -> None:
        # This method will be used to recover volumes from latent vectors using the model loaded with the previous
        # method (stored as self.model)
        # The method will not have any output. The generated volumes MUST be saved to a file following the previously
        # provided template stored in self.path_template
        pass
