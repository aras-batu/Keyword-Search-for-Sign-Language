from ignite.metrics import Metric
from ignite.exceptions import NotComputableError
from collections import defaultdict
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
import torch
import numpy as np
from metrics.utils.solution_utils import get_solution
from metrics.ineval import compute_f1
from models.model_utils import interpolate

class DictSolutionMetric(Metric):

    def __init__(self, gt, stack=True, output_transform=lambda x: x, device="cpu"):
        self.dict_time_videos = defaultdict(list)
        self.gt = gt
        self.stack = stack
        super(DictSolutionMetric, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self.dict_time_videos = defaultdict(list)
        super(DictSolutionMetric, self).reset()

    @reinit__is_reduced
    def update(self, y):
        y_pred, batch, size = y
        time_pred = interpolate(y_pred['time_out'], size//y_pred['time_out'].shape[1])
        
        for time, vid_name in zip(time_pred,batch["video_name"]):
            self.dict_time_videos[vid_name].append(time.detach().cpu())
            # self.dict_logits_videos[vid_name].append(logits.detach().cpu().unsqueeze(0).repeat(time.shape[0],1))
            # self.dict_target_videos[vid_name].append(tgt.detach().cpu())

    @sync_all_reduce("dict_time_videos")
    def compute(self):
        dict_res = {}
        
        
        for vn in self.dict_time_videos.keys():
            if self.stack:
                time = torch.vstack(self.dict_time_videos[vn]).float()
            else:
                time = self.dict_time_videos[vn]
            dict_res[vn] = {
                "time": time, 
            }
        # valid_gt = {}
        # dict_solution = get_solution(dict_res,ignore_class=self.num_classes, logit_threshold=0.0)
        # for k in list(dict_solution.keys()):
        #     valid_gt[k] = self.gt[k]
        # f1_score = compute_f1(dict_solution,valid_gt)
        return dict_res