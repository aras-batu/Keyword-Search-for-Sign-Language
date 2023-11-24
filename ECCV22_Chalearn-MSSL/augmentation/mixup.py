# Modified from source: https://github.com/hongyi-zhang/mixup/blob/master/cifar/utils.py
import torch
import numpy as np


def mixup_data(x, y, alpha=0.4, device="cuda"):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    # z_a, z_b = z, z[index]
    # return mixed_x, y_a, y_b, z_a, z_b, torch.tensor(lam).float()
    return mixed_x, y_a, y_b, torch.tensor(lam).float()




def mixup_criterion(criterion, pred, y_a, y_b, lam):
    y_a_loss = criterion(pred, y_a)
    # if type(y_a_loss) is tuple:
    #     y_a_loss = y_a_loss[0]

    y_b_loss = criterion(pred, y_b)
    # if type(y_b_loss) is tuple:
    #     y_b_loss = y_b_loss[0]

    return lam * y_a_loss + (1 - lam) * y_b_loss


def mixup_dict_criterion(criterion, pred, y_a, y_b, lam, additional_crit={}):
    y_a_loss, dict_a = criterion(pred, y_a, **additional_crit)
    if type(y_a_loss) is tuple:
        y_a_loss = y_a_loss[0]

    y_b_loss, dict_b = criterion(pred, y_b, **additional_crit)
    if type(y_b_loss) is tuple:
        y_b_loss = y_b_loss[0]

    for key, value in dict_b.items():
        dict_a[key] = (dict_a[key] + value) / 2
    return lam * y_a_loss + (1 - lam) * y_b_loss, dict_a
