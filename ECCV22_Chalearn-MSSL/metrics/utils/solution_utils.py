import numpy as np
import math
from itertools import groupby
import torch
from tqdm import tqdm
import copy

EPS = 1e-5


def beam_search_decoder(predictions, top_k=3):
    # start with an empty sequence with zero score
    output_sequences = [([], 0)]

    # looping through all the predictions
    for token_probs in predictions:
        new_sequences = []

        # append new tokens to old sequences and re-score
        for old_seq, old_score in output_sequences:
            for char_index in range(len(token_probs)):
                new_seq = old_seq + [char_index]
                # considering log-likelihood for scoring
                new_score = old_score + math.log(token_probs[char_index] + EPS)
                new_sequences.append((new_seq, new_score))

        # sort all new sequences in the de-creasing order of their score
        output_sequences = sorted(new_sequences, key=lambda val: val[1], reverse=True)

        # select top-k based on score
        # *Note- best sequence is with the highest score
        output_sequences = output_sequences[:top_k]

    return output_sequences


def compute_intervals(lst, ignore_class=60):
    list_of_intervals = []
    curr_index = 0
    for k, g in groupby(lst):
        end_index = curr_index + len([i for i in g])
        # TODO: POSSIBLY ADDED MIN SEQUENCE IGNORE LENGTH
        if k != ignore_class:
            if int(curr_index / 25 * 1000) != int((end_index - 1) / 25 * 1000):
                list_of_intervals.append(
                    [k, int(curr_index / 25 * 1000), int((end_index - 1) / 25 * 1000)]
                )
        curr_index = end_index
    return list_of_intervals


def get_solution(dict_pred, ignore_class, logit_threshold=0.5, stack_extra=True, apply_softmax=True):
    dict_solution = {}
    dict_predicted = copy.deepcopy(dict_pred)
    for vn in tqdm(dict_predicted.keys()):
        # logit_masks = dict_predicted[vn]['logits']>=logit_threshold
        # if stack_extra:
        #     logit_masks = torch.hstack([logit_masks,torch.ones(logit_masks.shape[0],1)])

        time_preds = dict_predicted[vn]["time"]
        # time_preds[~logit_masks.bool()]=-10
        if apply_softmax:
            predictions = torch.softmax((time_preds), axis=1)
        else:
            predictions = time_preds

        # TODO: update with better candidate selection
        candidates = beam_search_decoder(predictions, top_k=1)
        selected_candidate = candidates[0][0]

        list_of_intervals = compute_intervals(
            selected_candidate, ignore_class=ignore_class
        )
        dict_solution[vn] = list_of_intervals
    return dict_solution
