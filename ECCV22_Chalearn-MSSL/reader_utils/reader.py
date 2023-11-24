import cv2
import numpy as np
import torch
import copy
import pickle
import lmdb

def get_frames(vseq_data):
    key_frames = []
    for vi in vseq_data:
        frame = cv2.imdecode(
            np.frombuffer(vi, np.uint8),
            flags=cv2.IMREAD_COLOR,
        )
        frame = cv2.cvtColor(frame, code=cv2.COLOR_BGR2RGB)
        key_frames.append(frame)
    return key_frames

def transform_frames(frames, transform):
    transform = copy.deepcopy(transform)
    additional_targets = {}
    for i in range(1, len(frames)):
        additional_targets[f"image_{i}"] = "image"
    transform.add_targets(additional_targets)

    temp = {}

    for i, tmp in enumerate(frames):
        if i == 0:
            temp["image"] = tmp
        else:
            temp[f"image_{i}"] = tmp
    temp = transform(**temp)
    norm_frames = torch.stack(
        [temp[f"image_{ik}"] if ik != 0 else temp["image"] for ik in range(len(frames))]
    )

    return norm_frames


def get_data_point(lmdb_dir, key):
    with lmdb.open(
        path=lmdb_dir,
        readonly=True,
        readahead=False,
        lock=False,
        meminit=False,
    ) as env:
        with env.begin(write=False) as txn:
            vdata = pickle.loads(txn.get(key.encode("ascii")))
    return vdata