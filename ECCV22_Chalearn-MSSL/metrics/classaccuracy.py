from ignite.metrics import Metric
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
import torch
from models.model_utils import interpolate

EPSILON_FP16 = 1e-5


class ClassAccuracy(Metric):
    def __init__(
        self, threshold=0.5, k=1, num_classes=1000, output_transform=lambda x: x
    ):
        self.k = k
        self.correct = None
        self.total = None
        self.num_classes = num_classes
        self.threshold = threshold
        super(ClassAccuracy, self).__init__(output_transform=output_transform)

    @reinit__is_reduced
    def reset(self):
        self.correct = torch.zeros(self.num_classes)
        self.total = torch.zeros(self.num_classes)
        super(ClassAccuracy, self).reset()

    @reinit__is_reduced
    def update(self, output):
        y_pred, y = output
        y_pred = y_pred.detach().cpu().float()

        y_pred = interpolate(y_pred, y.shape[1] // y_pred.shape[1])
        y = y.detach().cpu()

        y = y == 1  # .reshape(-1)
        y = y.reshape(-1, self.num_classes)

        y_pred = torch.softmax(y_pred, axis=-1)
        y_pred = y_pred >= self.threshold  # .reshape(-1)
        y_pred = y_pred.reshape(-1, self.num_classes)

        for cl in range(self.num_classes):
            correct = y_pred[:, cl] == y[:, cl]
            actual_correct = correct[y[:, cl] == 1.0]
            self.correct[cl] += actual_correct.sum()
            self.total[cl] += actual_correct.shape[0]

        # max_logit = torch.topk(y_pred.detach().float().cpu(), k=self.k, dim=1)[1]
        # matches = torch.where(max_logit == y.unsqueeze(1))
        # correct_matches = max_logit[matches]

        # for cm in correct_matches:
        #     self.correct[cm] += 1
        # for y_i in y:
        #     self.total[y_i] += 1

    @sync_all_reduce("correct", "total")
    def compute(self):
        no_label_classes = self.total != 0
        return torch.mean(self.correct[no_label_classes] / self.total[no_label_classes])
