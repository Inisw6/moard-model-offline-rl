import importlib
import os
import pkgutil

_REGISTRY = {}


def register(key: str):
    def decorator(cls):
        _REGISTRY[key] = cls
        return cls

    return decorator


def make(key: str, **kwargs):
    if key not in _REGISTRY:
        available = ", ".join(_REGISTRY.keys())
        raise KeyError(
            f"No component registered under '{key}'. Available keys: {available}"
        )
    return _REGISTRY[key](**kwargs)


package_dir = os.path.dirname(__file__)
for finder, name, ispkg in pkgutil.iter_modules([package_dir]):
    importlib.import_module(f"{__package__}.{name}")
