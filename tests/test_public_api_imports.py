def test_public_imports() -> None:
    # A lightweight contract test: keep the most common imports stable.
    import yasfpy

    assert hasattr(yasfpy, "__version__")

    from yasfpy import YASF  # noqa: F401
