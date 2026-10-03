import pytest
import torch

from vesselwatch.models import build_model


@pytest.mark.parametrize("name", ["net", "net_max", "resnet18"])
@pytest.mark.parametrize("num_classes", [4, 25])
def test_output_shape(name, num_classes):
    model = build_model(name, num_classes).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, num_classes)


def test_legacy_state_dict_keys():
    """Saved weights of the original study must load into these classes."""
    net_keys = set(build_model("net", 4).state_dict())
    assert net_keys == {f"{layer}.{p}" for layer in ("conv1", "conv2", "conv3", "fc1") for p in ("weight", "bias")}

    max_keys = set(build_model("net_max", 4).state_dict())
    for layer in ("conv1a", "bn1b", "conv3c", "bn4c", "fc1", "fc2", "fc3"):
        assert f"{layer}.weight" in max_keys

    resnet = build_model("resnet101", 48)
    assert resnet.fc.weight.shape == (48, 2048)


def test_unknown_model_and_pretrained_custom():
    with pytest.raises(ValueError):
        build_model("nope", 4)
    with pytest.raises(ValueError):
        build_model("net", 4, pretrained=True)
