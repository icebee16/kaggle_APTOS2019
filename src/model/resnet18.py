from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models


def get_resnet18(task, weight=None, pretrained=False):
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
    model = models.resnet18(pretrained=False)
    if pretrained:
        model.load_state_dict(torch.load(Path(__file__).parents[2] / "model" / "pretrain" / "resnet18.pth"))

    num_features = model.fc.in_features
    if task == "classifier":
        model.fc = nn.Sequential(
            nn.Linear(in_features=num_features, out_features=5),
        )
    elif task == "regression":
        model.fc = nn.Sequential(
            nn.Linear(in_features=num_features, out_features=1)
        )
    else:
        print("{} task isn't implemented.".format(task))
        raise NotImplementedError

    if weight is not None:
        model.load_state_dict(weight)

    return model


if __name__ == "__main__":
    net = get_resnet18("classifier", pretrained=True)
    x = torch.randn(16, 3, 512, 512)
    y = net(x)
    assert torch.Size([16, 5]) == y.size()

    net = get_resnet18("regression", pretrained=True)
    x = torch.randn(16, 3, 512, 512)
    y = net(x)
    assert torch.Size([16, 1]) == y.size()

    print("complete")
