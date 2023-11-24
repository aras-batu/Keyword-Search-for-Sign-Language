import os
import argparse
import numpy as np
import evaluation_utils
import pickle

def create_folder(folder):
    print ('create_folder: {}'.format(folder))
    try:
        os.makedirs(folder)
        print("Directory " , folder ,  " Created ")
    except FileExistsError:
        print("Directory " , folder ,  " already exists")

def main(args):

    folder_in =args.input
    folder_out = args.output
    threshold_iou_min = args.threshold_iou_min
    threshold_iou_max = args.threshold_iou_max
    threshold_iou_step = args.threshold_iou_step
    if threshold_iou_max > threshold_iou_min:
        threshold_iou = np.arange(start=threshold_iou_min, stop=threshold_iou_max+(threshold_iou_step-0.0001), step=threshold_iou_step)
    else:
        threshold_iou = np.array([threshold_iou_min])

    print('folder_in: {}'.format(folder_in))
    print('folder_out: {}'.format(folder_out))
    print('threshold_iou: {}'.format(threshold_iou))

    # folder_out_metrics = os.path.join(folder_out, 'metrics')
    # create_folder(folder_out_metrics)

    ref_path = os.path.join(folder_in, 'ref')
    res_path = os.path.join(folder_in, 'res')

    with open(os.path.join(ref_path, 'ground_truth.pkl'), 'rb') as f:
        gt = pickle.load(f,encoding='bytes')

    with open(os.path.join(res_path, 'predictions.pkl'), 'rb') as f:
        p = pickle.load(f,encoding='bytes')

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

        data_performances = np.array([evaluation_utils.extract_performances(data_gt, data_p, iou_idx) for iou_idx in threshold_iou])
        for data_performances_idx in data_performances:
            tp = data_performances_idx[0]
            fp = data_performances_idx[1]
            fn = data_performances_idx[2]

            global_tp = global_tp + tp
            global_fp = global_fp + fp
            global_fn = global_fn + fn

    avg_precision, avg_recall, avg_f1 = evaluation_utils.calculate_metrics(global_tp, global_fp, global_fn)
    print ('********************************************************************************')
    print ('TOTAL_SAMPLES: {}'.format(len(gt.keys())))
    print (' -- global_tp: {}'.format(str(global_tp).replace('.',',')))
    print (' -- global_fp: {}'.format(str(global_fp).replace('.',',')))
    print (' -- global_fn: {}'.format(str(global_fn).replace('.',',')))

    print (' -- global_precision: {}'.format(str(avg_precision).replace('.',',')))
    print (' -- global_recall: {}'.format(str(avg_recall).replace('.',',')))
    print (' -- global_f1: {}'.format(str(avg_f1).replace('.',',')))
    print ('********************************************************************************')


    with open(os.path.join(folder_out, 'scores.txt'), 'w') as f:
        f.write('avg_f1:{}\n'.format(avg_f1))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate Result Output')
    parser.add_argument('--input', required=True, default='', type=str)
    parser.add_argument('--output', required=True, default='', type=str)
    parser.add_argument('--threshold_iou_min', required=False, default=0.2, type=float)
    parser.add_argument('--threshold_iou_max', required=False, default=0.8, type=float)
    parser.add_argument('--threshold_iou_step', required=False, default=0.05, type=float)
    arg = parser.parse_args()
    main(arg)



# python evaluate.py --input ./input --output ./output --threshold_iou_min 0.2 --threshold_iou_max 0.8 --threshold_iou_step 0.05
