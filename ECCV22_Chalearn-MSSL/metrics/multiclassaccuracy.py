from ignite.metrics import Metric
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
import torch

EPSILON_FP16 = 1e-5

class MultiClassAccuracy(Metric):
    def __init__(self, threshold=0.5, num_classes=1000, output_transform=lambda x: x):
        self.correct = None
        self.total = None
        self.num_classes = num_classes
        self.threshold = threshold
        super(MultiClassAccuracy, self).__init__(output_transform=output_transform)

    @reinit__is_reduced
    def reset(self):
        self.correct = torch.zeros(self.num_classes)
        self.total = torch.zeros(self.num_classes)
        super(MultiClassAccuracy, self).reset()

    @reinit__is_reduced
    def update(self, output):
        y_pred, y = output
        y = y.detach().cpu()
        
        y_p = y_pred.detach().float().cpu()>=self.threshold
        correct_matches = (y_p==y)

        for cm in range(correct_matches.shape[1]):
            self.correct[cm] += (correct_matches[:,cm] & (y[:,cm]==1.0)).sum()
            
            self.total[cm] += (y[:,cm]==1.0).sum()

    @sync_all_reduce("correct", "total")
    def compute(self):
        no_label_classes = self.total != 0
        return torch.mean(self.correct[no_label_classes] / self.total[no_label_classes])