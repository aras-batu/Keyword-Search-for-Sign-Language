import pickle
import numpy as np
import torch
from PIL import Image
from metrics.ineval import compute_f1

import torch
from tqdm import tqdm
import copy
from metrics.utils.solution_utils import beam_search_decoder, compute_intervals

def get_solution(dict_pred, ignore_class):
    dict_solution = {}
    dict_predicted = copy.deepcopy(dict_pred)
    for vn in tqdm(dict_predicted.keys()):

        predictions = dict_predicted[vn]
        predictions[-1,60] = 1

        # TODO: update with better candidate selection
        candidates = beam_search_decoder(predictions, top_k=1)
        selected_candidate = candidates[0][0]

        list_of_intervals = compute_intervals(
            selected_candidate, ignore_class=ignore_class
        )
        dict_solution[vn] = list_of_intervals
    return dict_solution

eps = 1e-8
num_classes = 60
window_size = 32


dict_weighting = {
    "x": 1.0,
    "x4": 1.0,
    "x8": 1.0,
    "x16": 1.0,
    "x32": 1.0,
}
total = 0
for k, v in dict_weighting.items():
    total += v


for k, v in dict_weighting.items():
    dict_weighting[k] = v / total
print(dict_weighting)


def get_results(res_path):
    dict_predicted = pickle.load(open(res_path, "rb"))

    dict_ens_predicted = {}
    for k, time_out in tqdm(dict_predicted.items()):
        predicted_time_out = torch.zeros(len(time_out["x"]) + window_size, 61)
        increment_counter = torch.zeros(len(time_out["x"]) + window_size)
        for key_time, time_value in time_out.items():
            tv = torch.softmax(time_value, axis=-1)
            new_time_value = torch.nn.functional.interpolate(
                tv.permute(0, 2, 1),
                size=time_out["x"].shape[1],
                mode="nearest"
            ).permute(0, 2, 1)
            for i, time in enumerate(new_time_value):
                predicted_time_out[i : i + window_size, :] += (
                    time * dict_weighting[key_time]
                )
                increment_counter[i : i + window_size] += 1 * dict_weighting[key_time]
        ens_time_out = predicted_time_out / (increment_counter.unsqueeze(1) + eps)
        dict_ens_predicted[k] = ens_time_out
    return dict_ens_predicted

from pathlib import Path
res_paths = Path("chalearn_results/test").glob("**/*.pkl")

dict_ens_predicted = {}
counter = 0
for res_path in tqdm(res_paths):
    if "_fold3_" in str(res_path):
        continue
    dict_res = get_results(res_path)
    if len(dict_ens_predicted)==0:
        dict_ens_predicted = dict_res
    else:
        list_of_keys = list(dict_res.keys())
        for key in list_of_keys:
            dict_ens_predicted[key]+=dict_res[key]
    counter+=1

for k, v in dict_ens_predicted.items():
    dict_ens_predicted[k] = v/counter

dict_solution = get_solution(dict_ens_predicted, ignore_class=num_classes)

with open('predictions.pkl', 'wb') as handle:
    pickle.dump(dict_solution, handle, protocol=4)