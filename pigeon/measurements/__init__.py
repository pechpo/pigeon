"""Register custom Habitat measurements for pigeon.

Importing this package registers the measures with Habitat's registry.
Call `ensure_registered()` before creating Env instances.
"""

# Side-effect imports to register measures
from . import exploration_map as _exploration_map  # noqa: F401
from . import traveled_stairs as _traveled_stairs  # noqa: F401
from . import traveled_distance as _traveled_distance  # noqa: F401


def ensure_registered() -> None:
    return None
