

from pathlib import Path
from tqdm.auto import tqdm

import pandas as pd
from PIL import Image, ImageDraw
import numpy as np
import io
import lmdb
from collections import defaultdict
import pickle
import shutil

from argparse import ArgumentParser
from io import BytesIO
import json

from scripts.video_utils import readVideo



n_bytes = 2**40
protocol = 4


tmp_lmdb_dir = "/tmp/lmdb"

parser = ArgumentParser(parents=[])


parser.add_argument(
    "--dest_dir",
    type=str,
    default="TEST/lmdb_videos/test",
)

parser.add_argument(
    "--dataset_dir",
    type=str,
    default="TEST/VIDEOS",
)

parser.add_argument(
    "--bbox_dir",
    type=str,
    default="data/track1/test_bboxes.json",
)

params, unknown = parser.parse_known_args()

dst_dir =  Path(params.dest_dir)
dataset_dir = Path(params.dataset_dir)
bbox_dir = params.bbox_dir


with open(
       params.bbox_dir, "r"
    ) as fp:
        data_bbox = json.load(fp)


env = lmdb.open(path=str(tmp_lmdb_dir), map_size=n_bytes)
txn = env.begin(write=True)

videos_for_lmdb = list((dataset_dir).glob("*.mp4"))
dict_of_keys = {}
for o, video_pth in enumerate(tqdm(videos_for_lmdb, leave=False)):
    video_name = video_pth.stem.replace(".cropped", "")

    bbox = data_bbox[video_name]

    box_width = bbox[2] - bbox[0]
    box_height = bbox[3] - bbox[1]

    margin_height = int(box_height * 0.1)
    margin_width = int(box_width * 0.1)

    frames, indices = readVideo(video_pth, interval="full")

    s, h, w, c = frames.shape

    start_width = max(bbox[0] - margin_width, 0)
    start_height = max(bbox[1] - margin_height, 0)

    end_width = min(bbox[2] + margin_width, w)
    end_height = min(bbox[3] + margin_height, h)

    frames = frames[:, start_height:end_height, start_width:end_width]

    counter = 0
    temps = []
    for i, frame in enumerate(frames):
        fr = Image.fromarray(frame)

        temp = BytesIO()
        fr.save(temp, format="jpeg")  # , subsampling=0, quality=90)
        temp.seek(0)
        temps.append(temp.read())
        counter += 1

    txn.put(
        key=f"{video_name}".encode("ascii"),
        value=pickle.dumps(temps, protocol=protocol),
        dupdata=False,
    )

    dict_of_keys[video_name] = {"sequence_length": counter}

    txn.commit()

    txn = env.begin(write=True)

txn.put(
    key=("details").encode("ascii"),
    value=pickle.dumps(dict_of_keys, protocol=protocol),
    dupdata=False,
)
txn.commit()

env.close()

if dst_dir.exists():
    shutil.rmtree(dst_dir)
shutil.move(f"{tmp_lmdb_dir}", f"{dst_dir}")
