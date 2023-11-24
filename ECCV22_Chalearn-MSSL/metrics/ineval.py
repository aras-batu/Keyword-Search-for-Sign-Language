import pickle

import metrics.evaluation_utils as evaluation_utils
import numpy as np


def compute_f1(p, gt):
    threshold_iou_min = 0.2
    threshold_iou_max = 0.8
    threshold_iou_step = 0.05
    if threshold_iou_max > threshold_iou_min:
        threshold_iou = np.arange(
            start=threshold_iou_min,
            stop=threshold_iou_max + (threshold_iou_step - 0.0001),
            step=threshold_iou_step,
        )
    else:
        threshold_iou = np.array([threshold_iou_min])

    global_tp = 0
    global_fp = 0
    global_fn = 0

    for fidx in gt.keys():
        data_gt = np.asarray(gt[fidx])
        # Si el usuario no proporciona resultado para un fichero de gt se estima que este es []
        try:
            data_p = np.asarray(p[fidx])
        except:
            data_p = np.array([])

        data_performances = np.array(
            [
                evaluation_utils.extract_performances(data_gt, data_p, iou_idx)
                for iou_idx in threshold_iou
            ]
        )
        for data_performances_idx in data_performances:
            tp = data_performances_idx[0]
            fp = data_performances_idx[1]
            fn = data_performances_idx[2]

            global_tp = global_tp + tp
            global_fp = global_fp + fp
            global_fn = global_fn + fn

    avg_precision, avg_recall, avg_f1 = evaluation_utils.calculate_metrics(
        global_tp, global_fp, global_fn
    )
    print(
        "********************************************************************************"
    )
    print("TOTAL_SAMPLES: {}".format(len(gt.keys())))
    print(" -- global_tp: {}".format(str(global_tp).replace(".", ",")))
    print(" -- global_fp: {}".format(str(global_fp).replace(".", ",")))
    print(" -- global_fn: {}".format(str(global_fn).replace(".", ",")))

    print(" -- global_precision: {}".format(str(avg_precision).replace(".", ",")))
    print(" -- global_recall: {}".format(str(avg_recall).replace(".", ",")))
    print(" -- global_f1: {}".format(str(avg_f1).replace(".", ",")))
    print(
        "********************************************************************************"
    )
    return avg_f1, avg_precision, avg_recall
