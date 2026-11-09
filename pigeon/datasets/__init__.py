"""Dataset registration helpers for pigeon."""


def ensure_registered() -> None:
    """Import dataset modules for Habitat registry side effects."""
    from . import ovon_object_nav_dataset  # noqa: F401
    from . import open_eqa_dataset  # noqa: F401
