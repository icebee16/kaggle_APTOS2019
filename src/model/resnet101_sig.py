from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models


def get_resnet101_sig(task, weight=None, pretrained=False):
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
    model = models.resnet101_sig(pretrained=False)
    if pretrained:
        model.load_state_dict(torch.load(Path(__file__).parents[2] / "model" / "pretrain" / "resnet101.pth"))

    num_features = model.fc.in_features
    if task == "classifier":
        model.fc = nn.Sequential(
            nn.Linear(in_features=num_features, out_features=5),
            nn.Sigmoid()
        )
    elif task == "regression":
        model.fc = nn.Sequential(
            nn.Linear(in_features=num_features, out_features=1),
            nn.Sigmoid()
        )
    else:
        print("{} task isn't implemented.".format(task))
        raise NotImplementedError

    if weight is not None:
        model.load_state_dict(weight)

    return model


if __name__ == "__main__":
    net = get_resnet101_sig("classifier", pretrained=True)
    x = torch.randn(16, 3, 512, 512)
    y = net(x)
    assert torch.Size([16, 5]) == y.size()

    net = get_resnet101_sig("regression", pretrained=True)
    x = torch.randn(16, 3, 512, 512)
    y = net(x)
    assert torch.Size([16, 1]) == y.size()

    print("complete")
