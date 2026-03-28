"""SmallModel - Compress large embedding models into small, fast students."""

from smallmodel.core import SmallModel
from smallmodel.teachers import TEACHERS, register_teacher

__version__ = "0.2.0"
__all__ = ["SmallModel", "TEACHERS", "register_teacher"]
