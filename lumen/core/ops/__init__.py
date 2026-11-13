"""Import every command module in this package (auto-discovered) so that the registry is populated.

Any ``*.py`` file placed in ``lumen/core/ops/`` whose name does not start with an underscore is
imported at startup; its ``@op`` functions register themselves. Modules are imported in
alphabetical order. User plugins in ``~/.lumen/plugins`` are loaded afterwards.

Plugin bookkeeping (used by Plugins ▸ Utilities ▸ Reload Plugins / Compile and Run):

``PLUGIN_OPS[file] = [op ids]``   every command a plugin file registered (filled by ``load_plugin_file``)
``PLUGIN_ERRORS[file] = message`` the last import error of a plugin file, if any
``reload_user_plugins()``         drop every plugin command and re-import all plugin files
"""
from __future__ import annotations

import importlib
import os
import pkgutil
import sys
from pathlib import Path

_FAILED: dict[str, str] = {}

PLUGIN_OPS: dict[str, list[str]] = {}
PLUGIN_ERRORS: dict[str, str] = {}
PLUGIN_MODULE_PREFIX = "lumen_plugin_"


def lumen_home() -> Path:
    """The user's Lumen folder: ``$LUMEN_HOME`` when set, else ``~/.lumen`` (shared with ``lumen.core.io``)."""
    from ..io import lumen_home as _home
    return _home()


def plugin_dirs() -> list[Path]:
    """``<lumen home>/plugins`` first, then every folder in ``$LUMEN_PLUGINS`` (``os.pathsep`` separated)."""
    dirs = [lumen_home() / "plugins"]
    if os.environ.get("LUMEN_PLUGINS"):
        dirs += [Path(p).expanduser() for p in os.environ["LUMEN_PLUGINS"].split(os.pathsep) if p.strip()]
    return dirs


def load_plugin_file(path: "str | os.PathLike[str]") -> list[str]:
    """(Re-)import one plugin file; returns the op ids it registered.

    Commands the same file registered earlier are removed first, so an edited plugin replaces itself
    instead of raising ``duplicate op id``.  Import errors propagate (and are kept in ``PLUGIN_ERRORS``);
    commands registered before the failure are still tracked so a later reload can drop them."""
    import importlib.util
    from ..registry import REGISTRY
    f = Path(path)
    key = str(f)
    for oid in PLUGIN_OPS.pop(key, []):
        REGISTRY.ops.pop(oid, None)
    PLUGIN_ERRORS.pop(key, None)
    before = set(REGISTRY.ops)
    name = PLUGIN_MODULE_PREFIX + f.stem
    try:
        spec = importlib.util.spec_from_file_location(name, f)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot import {f}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod  # dataclasses, pickling and introspection expect the module to be importable by name
        spec.loader.exec_module(mod)
    except Exception as e:
        PLUGIN_ERRORS[key] = f"{type(e).__name__}: {e}"
        raise
    finally:
        PLUGIN_OPS[key] = [oid for oid in REGISTRY.ops if oid not in before]
    return list(PLUGIN_OPS[key])


for _info in sorted(pkgutil.iter_modules(__path__), key=lambda m: m.name):
    if _info.name.startswith("_"):
        continue
    try:
        importlib.import_module(f"{__name__}.{_info.name}")
    except Exception as e:  # pragma: no cover – a broken module must not take the app down
        _FAILED[_info.name] = f"{type(e).__name__}: {e}"
        print(f"[lumen] command module {_info.name} failed to load: {e}", file=sys.stderr)


def load_user_plugins() -> list[str]:
    """Import every *.py in <lumen home>/plugins (and $LUMEN_PLUGINS) so that their @op commands register."""
    loaded = []
    for d in plugin_dirs():
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.py")):
            if f.name.startswith("_"):
                continue
            try:
                load_plugin_file(f)
                loaded.append(str(f))
            except Exception as e:  # pragma: no cover
                print(f"[lumen] plugin {f} failed: {e}", file=sys.stderr)
    return loaded


def reload_user_plugins() -> dict:
    """Remove every command that came from a plugin file and import all plugin files again.

    Returns ``{"files": [loaded paths], "ops": {file: [ids]}, "errors": {file: message}, "removed": [ids]}``."""
    from ..registry import REGISTRY
    removed: list[str] = []
    for ids in list(PLUGIN_OPS.values()):
        for oid in ids:
            if REGISTRY.ops.pop(oid, None) is not None:
                removed.append(oid)
    for oid, sp in list(REGISTRY.ops.items()):  # plugins imported before tracking existed (or by hand)
        if (getattr(sp.func, "__module__", "") or "").startswith(PLUGIN_MODULE_PREFIX):
            REGISTRY.ops.pop(oid, None)
            removed.append(oid)
    PLUGIN_OPS.clear()
    PLUGIN_ERRORS.clear()
    files = load_user_plugins()
    return {"files": files, "ops": {k: list(v) for k, v in PLUGIN_OPS.items()}, "errors": dict(PLUGIN_ERRORS), "removed": removed}



load_user_plugins()

# the commands of Fiji plugin jars installed from update sites (ops/fiji_plugins.py): registered after every bundled
# command so a plugin command whose name a Lumen command already has is labelled "… (Fiji plugin)" and Lumen's own
# (cross-validated) version stays the default for that name
try:
    from .fiji_plugins import register_installed as _register_fiji

    _register_fiji()
except Exception:  # noqa: BLE001 - a broken jar never stops Lumen from starting
    pass
