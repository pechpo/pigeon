import sys
import types


def install_vlfm_checkpoint_aliases() -> None:
    """Map legacy checkpoint module paths to pigeon modules during torch.load."""
    import pigeon.obs_transformers.resize as resize

    vlfm_pkg = sys.modules.get("vlfm")
    if vlfm_pkg is None:
        vlfm_pkg = types.ModuleType("vlfm")
        vlfm_pkg.__path__ = []  # type: ignore[attr-defined]
        sys.modules["vlfm"] = vlfm_pkg

    obs_pkg = sys.modules.get("vlfm.obs_transformers")
    if obs_pkg is None:
        obs_pkg = types.ModuleType("vlfm.obs_transformers")
        obs_pkg.__path__ = []  # type: ignore[attr-defined]
        sys.modules["vlfm.obs_transformers"] = obs_pkg

    setattr(vlfm_pkg, "obs_transformers", obs_pkg)
    setattr(obs_pkg, "resize", resize)
    sys.modules["vlfm.obs_transformers.resize"] = resize
