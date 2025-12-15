import torch

from irl.utils.torchops import as_tensor, ensure_2d, one_hot


def test_as_tensor_device_and_dtype():
    dev = torch.device("cpu")
    x = as_tensor([1, 2, 3], dev)
    assert torch.is_tensor(x) and x.device.type == "cpu" and x.dtype == torch.float32

    y = torch.tensor([1.0, 2.0], dtype=torch.float64)
    y2 = as_tensor(y, dev)
    assert y2.dtype == y.dtype


def test_ensure_2d_shapes():
    a = torch.randn(4)
    b = ensure_2d(a)
    assert b.shape == (1, 4)

    c = torch.randn(3, 4)
    d = ensure_2d(c)
    assert d.shape == (3, 4)

    e = torch.randn(2, 3, 5)
    f = ensure_2d(e)
    assert f.shape == (6, 5)


def test_one_hot_encoding():
    idx = torch.tensor([0, 2, 1])
    oh = one_hot(idx, 3)
    assert oh.shape == (3, 3)
    assert torch.allclose(oh.sum(dim=1), torch.ones(3))
    assert oh[0, 0] == 1 and oh[1, 2] == 1 and oh[2, 1] == 1
