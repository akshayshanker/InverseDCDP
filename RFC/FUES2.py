"""Compatibility shim exposing the FUES utilities from the sibling repo."""

# Re-export the numba-jitted helpers so existing imports keep working.
from .FUES import FUES, uniqueEG  # noqa: F401

__all__ = ["FUES", "uniqueEG"]
