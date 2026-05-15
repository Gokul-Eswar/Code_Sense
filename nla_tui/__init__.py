"""
NLA TUI Package
Real-time visualization of AI internal thoughts using Natural Language Autoencoders.
"""

from .interceptor import ThoughtInterceptor
from .client import NLAClient
from .registry import discover_nla_config

__all__ = ["ThoughtInterceptor", "NLAClient", "discover_nla_config"]
