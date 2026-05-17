"""
NLA TUI Package
Real-time visualization of AI internal thoughts using Natural Language Autoencoders.
"""

from .interceptor import ThoughtInterceptor
from .client import NLAClient
from .registry import discover_nla_config
from .wrapper import wrap_model, NLAWrapper

__all__ = ["ThoughtInterceptor", "NLAClient", "discover_nla_config", "wrap_model", "NLAWrapper"]

### ULTRA DETAILED CODE EXPLANATION ###
#
# This file serves as the package initializer for the `nla_tui` library.
#
# --- Package Documentation ---
# - The docstring at the top identifies the package as the "NLA TUI Package" and defines its purpose: 
#   providing real-time visualization of AI internal thoughts using Natural Language Autoencoders.
#
# --- Imports ---
# - `from .interceptor import ThoughtInterceptor`: Imports the `ThoughtInterceptor` class, which is responsible 
#   for hooking into model layers to capture internal activations.
# - `from .client import NLAClient`: Imports the `NLAClient` class, which handles communication with 
#   NLA Verbalizer models (either locally or via remote SGLang servers).
# - `from .registry import discover_nla_config`: Imports a utility function used to automatically find 
#   and load configuration mapping between base models and their corresponding NLA verbalizers.
# - `from .wrapper import wrap_model, NLAWrapper`: Imports `wrap_model` (a convenience function) and 
#   `NLAWrapper` (a class) used to wrap standard Hugging Face models to enable activation extraction.
#
# --- Variables ---
# - `__all__`: A list (type: `List[str]`) that defines the public API of the package. When a user performs 
#   `from nla_tui import *`, only the symbols listed here will be exported.
#   - `ThoughtInterceptor`: The activation capture engine.
#   - `NLAClient`: The verbalization interface.
#   - `discover_nla_config`: The configuration discovery tool.
#   - `wrap_model`: The model wrapping utility.
#   - `NLAWrapper`: The class used for model wrapping.
#
