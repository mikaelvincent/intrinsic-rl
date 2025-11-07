import pytest
import torch.nn as nn


def test_icm_mlp_legacy_import_shim():
    with pytest.warns(DeprecationWarning):
        from irl.intrinsic.icm import mlp  # legacy deep import alias
    net = mlp(4, (8, 8), out_dim=2)
    assert isinstance(net, nn.Sequential)


def test_proposed_regionstats_alias():
    from irl.intrinsic.proposed import _RegionStats
    with pytest.warns(DeprecationWarning):
        from irl.intrinsic.proposed import RegionStats  # alias to _RegionStats
    assert RegionStats is _RegionStats


def test_riac_export_diagnostics_alias():
    with pytest.warns(DeprecationWarning):
        from irl.intrinsic.riac import export_diagnostics  # alias to diagnostics.export_diagnostics
    assert callable(export_diagnostics)
