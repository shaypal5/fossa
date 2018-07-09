from .simple import LatestWindowAnomalyDetector  # noqa: F401
from .committee import (  # noqa: F401
    LastNWindowsAnomalyDetector,
)
import fossa.weights  # noqa: F401

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
