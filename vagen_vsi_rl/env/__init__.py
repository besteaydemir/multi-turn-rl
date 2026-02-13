from .common import Observation

# Lazy imports: each backend pulls its own heavy deps.
# Import them explicitly when needed, or use the convenience aliases below.


def _import_vsi():
    from .vsi_env import VSIEnv, EnvConfig
    return VSIEnv, EnvConfig


def _import_habitat():
    from .habitat_env import HabitatEnv, HabitatEnvConfig
    return HabitatEnv, HabitatEnvConfig


# Eagerly try to import both — but don't crash if deps are missing
try:
    from .vsi_env import VSIEnv, EnvConfig
except ImportError:
    VSIEnv = None  # type: ignore[assignment,misc]
    EnvConfig = None  # type: ignore[assignment,misc]

try:
    from .habitat_env import HabitatEnv, HabitatEnvConfig
except ImportError:
    HabitatEnv = None  # type: ignore[assignment,misc]
    HabitatEnvConfig = None  # type: ignore[assignment,misc]

__all__ = [
    "Observation",
    "VSIEnv",
    "EnvConfig",
    "HabitatEnv",
    "HabitatEnvConfig",
]
