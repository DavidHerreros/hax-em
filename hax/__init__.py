"""Hax — tools for studying conformational heterogeneity in CryoEM data.

Historically this module eagerly re-exported every subpackage::

    from .checkpointer import *
    from .generators import *
    ...

That flattened all public symbols into the ``hax`` namespace, but it also
imported JAX (and cuML/cupy) on *any* ``import hax`` — including the CLI
dispatcher's fast paths (``hax_project_manager -h`` / ``--version`` /
``update`` / unknown-program errors), which never touch a model. Reaching the
dispatcher alone cost ~10 s and a full JAX backend load.

We now defer those imports via PEP 562 ``__getattr__`` (Python >= 3.7): the
flattened top-level symbols are resolved lazily, on first access, by searching
the subpackages. This keeps the public ``import hax; hax.HetSIREN`` style API
working for notebooks while making the package root (and therefore
``hax.cli``) free of heavy imports until a model symbol is actually used.

Submodule imports are unaffected: ``from hax.utils import *`` and
``hax.networks.hetsiren`` import their subpackage directly and still load
whatever they need.
"""

import importlib

__version__ = '1.0.4'

# Subpackages previously flattened into the ``hax`` namespace via
# ``from .X import *``. Order is significant: it mirrors the original sequence
# of star-imports, where a later import shadowed an earlier one on a name
# collision. We therefore search in reverse so the *last* provider wins.
_SUBPACKAGES = (
    "checkpointer",
    "generators",
    "layers",
    "networks",
    "programs",
    "utils",
    "metrics",
    "viewers",
    "schedulers",
)

# Resolved flattened symbols, cached so repeated access does not re-search.
_attr_cache = {}


def __getattr__(name):
    """PEP 562 lazy attribute resolution for the flattened top-level API.

    Only invoked when normal attribute lookup on the module fails, so it costs
    nothing for already-imported submodules or eager names (``__version__``).
    """
    # Requesting a subpackage itself (e.g. ``hax.networks``): the eager
    # star-imports used to bind these as a side effect, so preserve that.
    if name in _SUBPACKAGES:
        return importlib.import_module(f".{name}", __name__)

    # ``from hax import *`` looks up ``__all__``; build it on demand (forces
    # every subpackage to import, exactly as the old eager module did).
    if name == "__all__":
        return _build_all()

    if name in _attr_cache:
        return _attr_cache[name]

    # Search subpackages in reverse (last star-import wins on collisions).
    for sub in reversed(_SUBPACKAGES):
        module = importlib.import_module(f".{sub}", __name__)
        if hasattr(module, name):
            value = getattr(module, name)
            _attr_cache[name] = value
            return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _build_all():
    names = []
    for sub in _SUBPACKAGES:
        module = importlib.import_module(f".{sub}", __name__)
        exported = getattr(module, "__all__", None)
        if exported is None:
            exported = [n for n in dir(module) if not n.startswith("_")]
        for n in exported:
            if n not in names:
                names.append(n)
    return names


def __dir__():
    """Expose the flattened names (forces subpackage imports) plus eager ones."""
    return sorted(set(globals()) | set(_build_all()))
