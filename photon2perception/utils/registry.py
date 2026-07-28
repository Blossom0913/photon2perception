"""
Lightweight component registry.

Provides a minimal, dependency-free registry mechanism (similar in spirit to
mmengine/mmdetection's Registry) so that models, datasets, losses and
transforms can be looked up by string name from YAML configs. This is the
backbone of the config-driven build system used by tools/train.py,
tools/eval.py and the export scripts — it lets us swap components (backbone
variants, loss functions, dataset types) purely through config files without
touching Python code.

Usage:
    MODELS = Registry('models')

    @MODELS.register_module()
    class RawViT(nn.Module):
        ...

    model = MODELS.build(dict(type='RawViT', embed_dim=384))
"""

from typing import Any, Callable, Dict, Optional, Type


class Registry:
    """A simple name -> class/function registry.

    Args:
        name: Human-readable registry name (used in error messages).
        parent: Optional parent registry to fall back to on lookup misses,
            allowing hierarchical registries (mirrors mmengine's design).
    """

    def __init__(self, name: str, parent: Optional["Registry"] = None):
        self._name = name
        self._module_dict: Dict[str, Any] = {}
        self._parent = parent

    def __len__(self) -> int:
        return len(self._module_dict)

    def __contains__(self, key: str) -> bool:
        return key in self._module_dict

    def __repr__(self) -> str:
        items = ', '.join(sorted(self._module_dict.keys()))
        return f"Registry(name='{self._name}', items=[{items}])"

    @property
    def name(self) -> str:
        return self._name

    @property
    def module_dict(self) -> Dict[str, Any]:
        return self._module_dict

    def get(self, key: str) -> Any:
        """Look up a registered class/function by name."""
        if key in self._module_dict:
            return self._module_dict[key]
        if self._parent is not None and key in self._parent:
            return self._parent.get(key)
        raise KeyError(
            f"'{key}' is not registered in registry '{self._name}'. "
            f"Available: {sorted(self._module_dict.keys())}"
        )

    def register_module(
        self,
        name: Optional[str] = None,
        force: bool = False,
    ) -> Callable:
        """Class/function decorator that registers the target under `name`.

        Args:
            name: Registration key. Defaults to the wrapped object's __name__.
            force: If True, silently overwrite an existing registration.
        """

        def _register(obj: Type) -> Type:
            key = name if name is not None else obj.__name__
            if not force and key in self._module_dict:
                raise KeyError(
                    f"'{key}' is already registered in registry '{self._name}'. "
                    f"Pass force=True to overwrite."
                )
            self._module_dict[key] = obj
            return obj

        return _register

    def build(self, cfg: Dict, **default_kwargs) -> Any:
        """Instantiate a registered class from a config dict.

        Args:
            cfg: Dict containing a 'type' key (registration name) plus any
                constructor kwargs. `cfg` itself is not mutated.
            default_kwargs: Extra kwargs merged in (cfg takes precedence).
        Returns:
            An instance built as `cls(**{**default_kwargs, **cfg_without_type})`.
        """
        if not isinstance(cfg, dict):
            raise TypeError(f"cfg must be a dict, got {type(cfg)}")
        cfg = dict(cfg)  # shallow copy — never mutate caller's config
        if 'type' not in cfg:
            raise KeyError(f"cfg must contain a 'type' key, got keys={list(cfg.keys())}")
        obj_type = cfg.pop('type')
        obj_cls = self.get(obj_type) if isinstance(obj_type, str) else obj_type
        kwargs = {**default_kwargs, **cfg}
        try:
            return obj_cls(**kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to build '{obj_type}' from registry '{self._name}' "
                f"with kwargs={kwargs}: {e}"
            ) from e


# ----- Global registries shared across the codebase -----

MODELS = Registry('models')          # backbones, necks, heads, full model wrappers
DATASETS = Registry('datasets')      # dataset classes
TRANSFORMS = Registry('transforms')  # data augmentation transforms
LOSSES = Registry('losses')          # loss functions
