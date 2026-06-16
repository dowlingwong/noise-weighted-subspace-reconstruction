"""GWOSC pipeline helpers.

These functions are intentionally lightweight until public-data dependencies
(`gwosc`, `gwpy`) are installed by the user.
"""

from .smoke import dependency_status

__all__ = ["dependency_status"]
