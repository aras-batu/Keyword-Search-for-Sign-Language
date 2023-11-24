import pickle

gt_train = pickle.load(open('chalearn/TRAIN/MSSL_TRAIN_SET_GT.pkl','rb'))
gt_valid = pickle.load(open('chalearn/VALIDATION/MSSL_VAL_SET_GT.pkl','rb'))

gt_trainvalid = {**gt_train, **gt_valid}

print(gt_trainvalid)

with open('chalearn/TRAINVALID/MSSL_TRAINVAL_SET_GT.pkl', 'wb') as handle:
    pickle.dump(gt_trainvalid, handle, protocol=4)