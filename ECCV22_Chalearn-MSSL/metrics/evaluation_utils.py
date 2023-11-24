import numpy as np

def extract_performances(data_gt, data_eval, threshold_iou):
    tp = fp = fn = 0
    if (data_gt.shape[0] == 0 and data_eval.shape[0] == 0):
        return tp, fp, fn
            
    if (data_gt.shape[0] == 0):
        fp = data_eval.shape[0]
        return tp, fp, fn

    if (data_eval.shape[0] == 0):
        fn = data_gt.shape[0]
        return tp, fp, fn
    
    # SORT DATA_EVAL BY START_TIME 
    data_eval = data_eval[np.argsort(data_eval[:,1])]
    # FILTER OVERLAP 
    data_eval_filtered = filter_overlap_interval(data_eval)
        
    for gt_idx in data_gt:
        class_id = gt_idx[0]
        data_gt_idx = np.delete(gt_idx, 0)            
        detections = 0
        
        data_eval_id = data_eval_filtered[data_eval_filtered[:,0]==class_id]
        for eval_idx in data_eval_id:
            data_eval_idx = np.delete(eval_idx, 0)
            if (get_temporal_iou_1d(data_eval_idx, data_gt_idx) >= threshold_iou):
                detections = detections + 1
        if detections > 0:
            tp = tp + 1
        
    fn = data_gt.shape[0] - tp
    fp = data_eval.shape[0] - tp

    return tp, fp, fn
    
    
def calculate_metrics(tp, fp, fn):
    
    if tp == 0:
        return 0, 0, 0
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = (2 * (precision * recall)) / (precision + recall)
        
    return precision, recall, f1

def get_temporal_iou_1d(v1, v2):
    earliest_start = min(v1[0],v2[0])
    latest_start = max(v1[0],v2[0])
    earliest_end = min(v1[1],v2[1])
    latest_end = max(v1[1],v2[1]) 
    iou = (earliest_end - latest_start) / (latest_end - earliest_start)
    return 0 if iou < 0 else iou

def get_data_sorted_start_time(data):
    data_out = np.copy(data)
    return data_out

def filter_overlap_interval(data):
    if data.shape[0] > 1:
        data_out = np.copy(data)
        data_true = np.array([ data[idx][0]!=data[idx+1][0] or data[idx][2]<=data[idx+1][1] for idx in range(data.shape[0]-1)])
        data_true = np.append(data_true, True)
        #print(data_true)
        return data_out[data_true]
    return data



if __name__ == '__main__':
    
    gt = np.array([[20,3800,4680],
                   [70,13400,13960],
                   [20,14680,15360],
                   [20,15400,16000],
                   [20,31760,32520],
                   [70,39160,39840],
                   [20,41760,42600],
                   [70,51720,52200],
                   [20,53120,53880],
                   [20,53880,54280],
                   [20,61160,61960],
                   [20,64640,65440],
                   [70,67440,68280],
                   [64,68920,69520],
                   [20,70600,71280],
                   [20,73800,74360],
                   [20,80080,80760],
                   [20,88360,88800],
                   [20,94480,95120],
                   [20,101360,102080],
                   [20,106960,107680],
                   [20,109000,109640],
                   [71,111920,112240],
                   [36,115000,115760],
                   [20,127840,128560]])
    
    gt2 = np.array([[20,3800,4680],
                   [70,13400,13960],
                   [70,51720,52200],
                   [20,15400,16000],
                   [71,111920,112240],
                   [70,39160,39840],
                   [20,41760,42600],
                   [20,53120,53880],
                   [20,53880,54280],
                   [20,61160,61960],
                   [20,64640,65440],
                   [70,67440,68280],
                   [64,68920,69520],
                   [20,70600,71280],
                   [20,73800,74360],
                   [20,80080,80760],
                   [20,88360,88800],
                   [20,94480,95120],
                   [20,14680,15360],
                   [20,31760,32520],
                   [20,101360,102080],
                   [20,106960,107680],
                   [20,109000,109640],
                   [36,115000,115760],
                   [20,127840,128560]])
    
    print (gt.shape)
    print(extract_performances(gt, gt2, 0.5))
