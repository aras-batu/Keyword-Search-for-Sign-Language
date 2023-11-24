from torch.utils.data import Dataset, DataLoader
import lmdb
import pandas as pd
import pickle
import numpy as np
import cv2
import torch


def prep_overlap(b):
    def getOverlap(a):
        return max(0, min(a[1], b[1]) - max(a[0], b[0])) > 0

    return getOverlap


class ContinuousIsolatedDataset(Dataset):
    def __init__(
        self,
        lmdb_dir,
        csv_dir,
        transform,
        seq_len,
        num_classes,
        scores=False,
        neg_prob=0.0,
    ):
        super(ContinuousIsolatedDataset, self).__init__()

        self.lmdb_dir = lmdb_dir

        self.df = pd.read_csv(csv_dir)

        self.lmdb = None

        self.env = None
        self.seq_len = seq_len
        self.transform = transform
        if transform:
            additional_targets = {}
            for i in range(1, seq_len):
                additional_targets[f"image_{i}"] = "image"
            self.transform.add_targets(additional_targets)
        self.num_classes = num_classes
        self.scores = scores
        self.neg_prob = neg_prob

    def __len__(self):
        return len(self.df)

    def get_selection(self, item):
        if np.random.random() < self.neg_prob:

            start_index = np.random.randint(0, item["fcount"] - self.seq_len - 1)
            end_index = start_index + self.seq_len
        else:
            start, end = item["start_frame"], item["end_frame"]
            rng_start, rng_end = max(start - self.seq_len // 2, 0), min(
                end + self.seq_len // 2, item["fcount"]
            )
            if rng_end - self.seq_len > rng_start:
                start_index = np.random.randint(rng_start, rng_end - self.seq_len)
            else:
                start_index = rng_end - self.seq_len

            end_index = start_index + self.seq_len
        return start_index, end_index

    def get_video_selection(self, item, index):
        data = pickle.loads(self.txn.get(item["video"].encode("ascii")))

        start_index, end_index = self.get_selection(item)
        frames = []
        for i in range(start_index, end_index):
            frame = cv2.imdecode(
                np.frombuffer(data[i], np.uint8),
                flags=cv2.IMREAD_COLOR,
            )
            frame = cv2.cvtColor(frame, code=cv2.COLOR_BGR2RGB)
            frames.append(frame)

        return frames, np.arange(start_index, end_index)

    def prepare_item(self, item, index):
        frames, selection = self.get_video_selection(item, index)

        df_vid = self.df[self.df.video == item["video"]]
        # df_select = df_vid[(df_vid.start_frame.isin(selection)) & (df_vid.end_frame.isin(selection))]

        b = [selection[0], selection[-1]]

        df_select = df_vid[
            df_vid[["start_frame", "end_frame"]].apply(prep_overlap(b), axis=1)
        ]

        # display(df_select)
        # print(selection)

        targets = torch.zeros(self.seq_len, self.num_classes + 1)
        for index, row in df_select.iterrows():
            s = max(row["start_frame"] - selection[0], 0)
            duration = row["end_frame"] - row["start_frame"] + 1

            e = min(row["start_frame"] - selection[0] + duration, self.seq_len)
            targets[s:e, row["label"]] = 1.0
        targets[:, -1] = (targets.sum(axis=1) == 0.0).float()

        logit_targets = (targets.sum(axis=0) >= 1)[:-1].float()

        if self.transform:
            temp = {}

            for i, tmp in enumerate(frames):
                if i == 0:
                    temp["image"] = tmp
                else:
                    temp[f"image_{i}"] = tmp
            temp = self.transform(**temp)
            frames = torch.stack(
                [
                    temp[f"image_{ik}"] if ik != 0 else temp["image"]
                    for ik in range(self.seq_len)
                ]
            )
        else:
            frames = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)

        return {
            "index": index,
            "frames": frames.float(),
            "targets": targets,
            "logit_targets": logit_targets,
        }

    def __getitem__(self, index: int):

        if not self.env:
            self.env = lmdb.open(
                path=self.lmdb_dir,
                readonly=True,
                readahead=False,
                lock=False,
                meminit=False,
            )
            self.txn = self.env.begin(write=False)

        item = self.df.iloc[index]

        res = self.prepare_item(item, index)
        if self.scores:
            return {
                "index": torch.tensor(index),
                "frames": res["frames"],
                "targets": res["targets"],
                "logit_targets": res["logit_targets"],
                "score": torch.tensor(item["score"], requires_grad=False),
            }
        else:
            return {
                "index": torch.tensor(index),
                "frames": res["frames"],
                "targets": res["targets"],
                "logit_targets": res["logit_targets"],
                # "item":item
            }

