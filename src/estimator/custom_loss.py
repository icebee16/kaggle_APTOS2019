import torch
from torch.nn import Module

from sklearn.metrics import cohen_kappa_score


class WeightedMSELoss(Module):
    def __init__(self, weight):
        super(WeightedMSELoss, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.weight = torch.tensor(weight, device=self.device, dtype=torch.float)

    def forward(self, preds, target):
        weight_list = torch.zeros(target.size(), device=self.device, dtype=torch.float)
        for i in range(self.weight.size()[0]):
            weight_list[target[:] == i] = self.weight[i]

        return torch.sum(weight_list * (preds - target) ** 2) / target.size()[0]


class InverseQWKLoss(Module):
    def __init__(self):
        super(InverseQWKLoss, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, preds, target):
        qwk_score = cohen_kappa_score(torch.round(preds).cpu().detach().numpy(),
                                      target.cpu().detach().numpy(),
                                      weights="quadratic")
        return torch.tensor(-qwk_score, device=self.device)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float
    c = WeightedMSELoss([.1, .2, .3, .4, .5])

    pred = torch.tensor([1, 2, 3, 4, 5], device=device, dtype=dtype)
    target = torch.tensor([0, 1, 2, 3, 4], device=device, dtype=dtype)
    print(c.forward(pred, target))
