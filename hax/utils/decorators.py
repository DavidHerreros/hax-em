import functools
import inspect
from flax import nnx


def save_config(init_func):
    """Decorator to automatically save __init__ arguments and class path."""

    @functools.wraps(init_func)
    def wrapper(self, *args, **kwargs):
        # 1. Bind the passed arguments to the function signature
        sig = inspect.signature(init_func)
        bound_args = sig.bind(self, *args, **kwargs)
        bound_args.apply_defaults()

        # 2. Extract arguments, filtering out 'self' and 'rngs'
        config_dict = {}
        for k, v in bound_args.arguments.items():
            if k not in ('self', 'rngs'):
                if isinstance(v, list):
                    v = nnx.List(v)
                elif isinstance(v, dict):
                    v = nnx.Dict(v)
                config_dict[k] = v

        # 3. INJECT THE IMPORT PATH
        # Gets the module (e.g., 'vision_models') and class (e.g., 'CNNModel')
        config_dict['_target_'] = f"{self.__class__.__module__}.{self.__class__.__name__}"

        # 4. Save to the instance
        self.config = nnx.Dict(config_dict)

        # 5. Call the original __init__
        return init_func(self, *args, **kwargs)

    return wrapper