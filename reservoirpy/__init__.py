# Licence: MIT License
# Copyright: Xavier Hinaut (2018) <xavier.hinaut@inria.fr>

import atexit
import importlib
import os
import shutil
import tempfile
import uuid

from . import activationsfunc, hyper, mat_gen, nodes, observables, type
from ._version import __version__
from .esn import ESN
from .model import Model
from .node import Node
from .utils.random import set_seed
from .utils.save_load import load, save

# Unique temporary directory, to prevent permission errors on shared machines.
# It is only created when something is written to it (see reservoirpy.datasets)
# and is removed when the interpreter exits.
_TEMPDIR = os.path.join(tempfile.gettempdir(), f"reservoirpy-temp-{uuid.uuid4()}")
_TEMPDIR_OWNER = os.getpid()


def _remove_tempdir():  # pragma: no cover
    # forked workers inherit this handler: only the owner cleans up
    if os.getpid() == _TEMPDIR_OWNER:
        shutil.rmtree(_TEMPDIR, ignore_errors=True)


atexit.register(_remove_tempdir)

__all__ = [
    "__version__",
    "mat_gen",
    "activationsfunc",
    "observables",
    "hyper",
    "datasets",
    "nodes",
    "Node",
    "Model",
    "set_seed",
    "ESN",
    "type",
    "save",
    "load",
]


def __getattr__(attr):
    if attr in ["experimental", "verbosity", "compat"]:
        raise type.DeprecatedError(f"reservoirpy.{attr} has been removed in ReservoirPy v0.4.0.")
    elif attr == "datasets":
        # imported on first access: it sets up an on-disk cache
        return importlib.import_module(".datasets", __name__)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {attr!r}")
