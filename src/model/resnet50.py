from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models


def get_resnet50(task, weight=None, pretrained=False):
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
    resnet50 = models.resnet50(pretrained=False)
    if pretrained:
        resnet50.load_state_dict(torch.load(Path(__file__).parents[2] / "model" / "pretrain" / "resnet50.pth"))

    if task == "classifier":
        resnet50.fc = nn.Sequential(
            nn.Linear(in_features=2048, out_features=5),
            nn.Sigmoid()
        )
    elif task == "regression":
        resnet50.fc = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1)
        )
    else:
        print("{} task isn't implemented.".format(task))
        raise NotImplementedError

    if weight is not None:
        resnet50.load_state_dict(weight)

    return resnet50


if __name__ == "__main__":
    net = get_resnet50("classifier", pretrained=True)
    x = torch.randn(16, 3, 512, 512)
    y = net(x)
    assert torch.Size([16, 5]) == y.size()

    net = get_resnet50("regression", pretrained=True)
    x = torch.randn(16, 3, 512, 512)
    y = net(x)
    assert torch.Size([16, 1]) == y.size()

    print("complete")
