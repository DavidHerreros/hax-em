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

        # 2. Extract arguments, filtering out 'self', 'rngs' and any variadic
        #    (*args / **kwargs) parameters. The catch-all is used only to absorb
        #    deprecated arguments from older configs; persisting it would store an
        #    empty {} that nests one level deeper on every save/load round-trip.
        variadic = {name for name, p in sig.parameters.items()
                    if p.kind in (inspect.Parameter.VAR_POSITIONAL,
                                  inspect.Parameter.VAR_KEYWORD)}
        config_dict = {}
        for k, v in bound_args.arguments.items():
            if k not in ('self', 'rngs') and k not in variadic:
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