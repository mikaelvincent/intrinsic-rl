def test_import_and_version():
    import irl
    assert hasattr(irl, "__version__")
    assert isinstance(irl.__version__, str)
    assert len(irl.__version__) > 0
