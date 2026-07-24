"""RNS-LLM experimental CUDA package.

PyTorch C++/CUDA extension modules link against libraries such as ``libc10`` and
``libtorch_python``.  Importing :mod:`torch` first loads those DSOs, which makes
direct imports such as ``import rns_llm._PREFILL`` reliable in a fresh Python
process and in Google Colab.  The CUDA extensions themselves are still not
imported eagerly, so the pure-Python audit utilities remain usable without a
compiled extension.
"""

try:
    import torch as _torch  # noqa: F401  # preload PyTorch shared libraries
except Exception:
    _torch = None

__version__ = "0.14.2"
