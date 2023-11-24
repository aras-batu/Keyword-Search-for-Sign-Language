from ignite.metrics import Metric
from ignite.exceptions import NotComputableError
from collections import defaultdict
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
import torch
import numpy as np
from metrics.utils.solution_utils import get_solution
from metrics.ineval import compute_f1

class SolutionTestMetric(Metric):

    def __init__(self, output_transform=lambda x: x, device="cpu"):
        self.dict_time_videos = defaultdict(list)
        # self.dict_logits_videos = defaultdict(list)
        self.dict_target_videos = defaultdict(list)
        super(SolutionTestMetric, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self.dict_time_videos = defaultdict(list)
        # self.dict_logits_videos = defaultdict(list)
        self.dict_target_videos = defaultdict(list)
        super(SolutionTestMetric, self).reset()

    @reinit__is_reduced
    def update(self, y):
        y_pred, batch = y
        for time, tgt, vid_name in zip(y_pred['time_out'],batch['targets'],batch["video_name"]):
            self.dict_time_videos[vid_name].append(time.detach().cpu())
            self.dict_target_videos[vid_name].append(tgt.detach().cpu())

    @sync_all_reduce("dict_time_videos", "dict_target_videos")
    def compute(self):
        dict_res = {}
        
        
        for vn in self.dict_time_videos.keys():
            dict_res[vn] = {
                "time": torch.vstack(self.dict_time_videos[vn]).float(), 
                "targets": torch.vstack(self.dict_target_videos[vn]).float()
            }
        return dict_res