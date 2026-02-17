"""Arc Fabric — checkpoint / fork / rewind platform for DreamZero."""

__all__ = ["WorldStateSnapshot", "StateManager"]

# Guard heavy imports so the package is importable without torch/GPU
# (needed for mock-mode server and CLI tools).
try:
    from .state import WorldStateSnapshot
    from .manager import StateManager
except ImportError:
    pass
