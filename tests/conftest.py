import pytest
import torch


def _get_devices():
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


@pytest.fixture(params=_get_devices())
def device(request):
    return torch.device(request.param)


@pytest.fixture(params=[torch.float32, torch.float64], ids=["f32", "f64"])
def dtype(request):
    return request.param


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(42)
