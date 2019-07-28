from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models


def get_resnet50(task, weight=None, is_train=True):
    """
    Parameters
    ----------
    task: str
        kind of task.
    weight: torch.dict
        pretrained weight in training section.
    is_train: bool
        called from train part or test part.
    """
    resnet50 = models.resnet50(pretrained=False)
    if is_train:
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

    return resnet50


if __name__ == "__main__":
    net = get_resnet50("classifier", is_train=True)
    x = torch.randn(16, 3, 512, 512)
    y = net(x)
    assert torch.Size([16, 5]) == y.size()

    net = get_resnet50("regression", is_train=True)
    x = torch.randn(16, 3, 512, 512)
    y = net(x)
    assert torch.Size([16, 1]) == y.size()

    print("complete")
