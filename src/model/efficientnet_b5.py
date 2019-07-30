from pathlib import Path

import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet


def get_efficientnet_b5(task, weight=None, pretrained=False):
    """
    Parameters
    ----------
    task: str
        kind of task.
    weight: torch.dict
        pretrained weight in training section.
        so, this parameter use in inference.
    pretrained: bool
        load torchvision model weight.
    """
    model = EfficientNet.from_name("efficientnet-b5")
    if pretrained:
        model.load_state_dict(torch.load(Path(__file__).parents[2] / "model" / "pretrain" / "efficientnet-b5.pth"))

    num_features = model.classifier.in_features
    if task == "classifier":
        model._fc = nn.Sequential(
            nn.Linear(in_features=num_features, out_features=5),
        )
    elif task == "regression":
        model._fc = nn.Sequential(
            nn.Linear(in_features=num_features, out_features=1)
        )
    else:
        print("{} task isn't implemented.".format(task))
        raise NotImplementedError

    if weight is not None:
        model.load_state_dict(weight)

    return model


if __name__ == "__main__":
    net = get_efficientnet_b5("classifier", pretrained=True)
    x = torch.randn(16, 3, 512, 512)
    y = net(x)
    assert torch.Size([16, 5]) == y.size()

    net = get_efficientnet_b5("regression", pretrained=True)
    x = torch.randn(16, 3, 512, 512)
    y = net(x)
    assert torch.Size([16, 1]) == y.size()

    print("complete")
