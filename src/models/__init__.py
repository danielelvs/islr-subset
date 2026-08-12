from __future__ import annotations

"""Model package.

Optional timm-based models are imported lazily by ``BaseModel.get_by_name`` so
ResNet experiments do not require timm merely to import the package.
"""

from models.base_model import BaseModel

__all__ = ["BaseModel"]
