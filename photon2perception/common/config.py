"""
Config loading utilities.

A thin wrapper around YAML/JSON loading that supports:
- `_base_`: inherit from one or more base config files (mmcv/mmengine style),
  merged recursively so experiment configs only need to specify overrides.
- Dotted-key overrides from the CLI (e.g. `model.embed_dim=512`), which is
  how tools/train.py, tools/eval.py and tools/export.py let users tweak a
  config without editing YAML files.
- Attribute-style access (`cfg.model.embed_dim`) in addition to dict access,
  purely for convenience.

Design goal: zero required third-party dependencies beyond PyYAML (already
in requirements.txt), so this works identically on a laptop and on a
restricted edge-device build machine.
"""

import copy
import json
import os
from typing import Any, Dict, List, Optional, Union

import yaml


class ConfigDict(dict):
    """A dict that also supports attribute access, recursively.

    Only used as a convenience view over plain dicts — all downstream code
    (registry.build, json.dump, yaml.dump) still sees a plain dict/ConfigDict
    which is a dict subclass, so no special-casing is required elsewhere.
    """

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as e:
            raise AttributeError(key) from e
        if isinstance(value, dict) and not isinstance(value, ConfigDict):
            value = ConfigDict(value)
            self[key] = value
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError as e:
            raise AttributeError(key) from e


def _to_config_dict(obj: Any) -> Any:
    """Recursively convert nested dicts into ConfigDict for attribute access."""
    if isinstance(obj, dict):
        return ConfigDict({k: _to_config_dict(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_config_dict(v) for v in obj]
    return obj


def _merge_dict(base: Dict, override: Dict) -> Dict:
    """Recursively merge `override` into `base`, returning a new dict.

    Lists are replaced wholesale (not concatenated) — this matches the
    common expectation that overriding a list-valued field (e.g.
    `img_size: [512, 512]`) replaces it rather than appending.
    """
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_raw_file(path: str) -> Dict:
    ext = os.path.splitext(path)[1].lower()
    with open(path, 'r') as f:
        if ext in ('.yaml', '.yml'):
            data = yaml.safe_load(f) or {}
        elif ext == '.json':
            data = json.load(f)
        else:
            raise ValueError(f"Unsupported config extension '{ext}' for {path}")
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a top-level mapping")
    return data


def load_config(path: str, _visited: Optional[List[str]] = None) -> ConfigDict:
    """Load a YAML/JSON config file, resolving `_base_` inheritance.

    A config may specify:
        _base_: base.yaml               # single base file
        _base_: [base1.yaml, base2.yaml] # multiple base files (later wins)

    Base paths are resolved relative to the file that references them.

    Args:
        path: Path to the config file.
        _visited: Internal recursion guard against circular `_base_` refs.
    Returns:
        A merged ConfigDict (base config overridden by this file's contents).
    """
    path = os.path.abspath(path)
    _visited = _visited or []
    if path in _visited:
        raise ValueError(f"Circular config inheritance detected: {' -> '.join(_visited + [path])}")
    _visited = _visited + [path]

    data = _load_raw_file(path)
    bases = data.pop('_base_', None)

    if bases is None:
        merged: Dict = {}
    else:
        if isinstance(bases, str):
            bases = [bases]
        base_dir = os.path.dirname(path)
        merged = {}
        for base_rel in bases:
            base_path = base_rel if os.path.isabs(base_rel) else os.path.join(base_dir, base_rel)
            base_cfg = load_config(base_path, _visited=_visited)
            merged = _merge_dict(merged, dict(base_cfg))

    merged = _merge_dict(merged, data)
    return _to_config_dict(merged)


def _cast_value(raw: str) -> Any:
    """Best-effort cast of a CLI override string to bool/int/float/None/str."""
    lowered = raw.lower()
    if lowered in ('true', 'false'):
        return lowered == 'true'
    if lowered in ('null', 'none'):
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    # Try YAML for lists/dicts written inline, e.g. "[512, 512]"
    try:
        parsed = yaml.safe_load(raw)
        if isinstance(parsed, (list, dict)):
            return parsed
    except yaml.YAMLError:
        pass
    return raw


def apply_cli_overrides(cfg: ConfigDict, overrides: Optional[List[str]]) -> ConfigDict:
    """Apply `key.path=value` CLI overrides onto a loaded config.

    Example:
        apply_cli_overrides(cfg, ['model.embed_dim=512', 'training.epochs=10'])

    Args:
        cfg: Config loaded via `load_config`.
        overrides: List of 'dotted.key=value' strings (e.g. from argparse
            `nargs='+'`). None or empty list is a no-op.
    Returns:
        The same cfg object, mutated in place, for convenience chaining.
    """
    if not overrides:
        return cfg
    for item in overrides:
        if '=' not in item:
            raise ValueError(f"Invalid override '{item}', expected 'key.path=value'")
        key_path, raw_value = item.split('=', 1)
        value = _cast_value(raw_value)
        keys = key_path.split('.')
        node: Dict = cfg
        for k in keys[:-1]:
            if k not in node or not isinstance(node[k], dict):
                node[k] = ConfigDict()
            node = node[k]
        node[keys[-1]] = value
    return cfg


def save_config(cfg: Union[ConfigDict, Dict], path: str) -> None:
    """Dump a (possibly merged/overridden) config to disk for reproducibility."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    plain = json.loads(json.dumps(cfg))  # strip ConfigDict wrapper -> plain dict
    ext = os.path.splitext(path)[1].lower()
    with open(path, 'w') as f:
        if ext == '.json':
            json.dump(plain, f, indent=2)
        else:
            yaml.safe_dump(plain, f, sort_keys=False)
