from .simple import LatestWindowAnomalyDetector  # noqa: F401
from .committee import (  # noqa: F401
    LastNWindowsAnomalyDetector,
)
from .condition_committee import (  # noqa: F401
    __load_condition_builder_attributes__,
    ConditionsCommitteeAnomalyDetector,
)
import fossa.weights  # noqa: F401
# import fossa.conditions  # noqa: F401
from . import conditions  # noqa: F401


from ._version import get_versions
__version__ = get_versions()['version']
del get_versions


__load_condition_builder_attributes__(conditions)
for name in [
    '__load_condition_builder_attributes__',
    '_version', 'get_versions',
]:
    try:
        globals().pop(name)
    except KeyError:
        pass
try:
    del name  # pylint: disable=W0631
except NameError:
    pass
