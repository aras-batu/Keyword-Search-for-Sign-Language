from ignite.metrics import Metric
from ignite.exceptions import NotComputableError
from collections import defaultdict
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
import torch
import numpy as np
from metrics.utils.solution_utils import get_solution
from metrics.ineval import compute_f1

from models.model_utils import interpolate


class SolutionMetric(Metric):
    def __init__(
        self,
        gt,
        num_classes,
        output_transform=lambda x: x,
        device="cpu",
        apply_sigmoid=True,
    ):
        self.dict_time_videos = defaultdict(list)
        # self.dict_logits_videos = defaultdict(list)
        self.dict_target_videos = defaultdict(list)
        self.apply_sigmoid = apply_sigmoid
        self.num_classes = num_classes
        self.gt = gt
        super(SolutionMetric, self).__init__(
            output_transform=output_transform, device=device
        )

    @reinit__is_reduced
    def reset(self):
        self.dict_time_videos = defaultdict(list)
        # self.dict_logits_videos = defaultdict(list)
        self.dict_target_videos = defaultdict(list)
        super(SolutionMetric, self).reset()

    @reinit__is_reduced
    def update(self, y):
        y_pred, batch = y

        time_pred = interpolate(
            y_pred["time_out"], batch["targets"].shape[1] // y_pred["time_out"].shape[1]
        )

        for time, tgt, vid_name in zip(
            time_pred, batch["targets"], batch["video_name"]
        ):

            self.dict_time_videos[vid_name].append(time.detach().cpu())
            # self.dict_logits_videos[vid_name].append(logits.detach().cpu().unsqueeze(0).repeat(time.shape[0],1))
            self.dict_target_videos[vid_name].append(tgt.detach().cpu())

    @sync_all_reduce("dict_time_videos", "dict_target_videos")
    def compute(self):
        dict_res = {}

        for vn in self.dict_time_videos.keys():
            dict_res[vn] = {
                # "logits": torch.sigmoid(torch.vstack(self.dict_logits_videos[vn]).float()) if self.apply_sigmoid else torch.vstack(self.dict_logits_videos[vn]).float(),
                "time": torch.vstack(self.dict_time_videos[vn]).float(),
                "targets": torch.vstack(self.dict_target_videos[vn]).float(),
            }
        valid_gt = {}
        dict_solution = get_solution(
            dict_res, ignore_class=self.num_classes, logit_threshold=0.0
        )
        for k in list(dict_solution.keys()):
            valid_gt[k] = self.gt[k]
        f1_score, avg_precision, avg_recall = compute_f1(dict_solution, valid_gt)

        return {"f1": f1_score, "precision": avg_precision, "recall": avg_recall}
