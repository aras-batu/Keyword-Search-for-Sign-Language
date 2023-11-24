from torch.utils.data import Dataset, DataLoader
import lmdb
import pandas as pd
import pickle
import numpy as np
import cv2
import torch
from collections import defaultdict

def prep_overlap(b):
    def getOverlap(a):
        return max(0, min(a[1], b[1]) - max(a[0], b[0]))>0
    return getOverlap


class ContinuousDataset(Dataset):
    def __init__(
        self,
        lmdb_dir,
        csv_dir,
        transform,
        seq_len,
        num_classes
    ):
        super(ContinuousDataset, self).__init__()

      
        self.lmdb_dir = lmdb_dir
        
        self.df = pd.read_csv(csv_dir)
        
        self.lmdb = None

        self.env = None
        self.seq_len = seq_len
        self.transform = transform 
        if transform:
            additional_targets={}
            for i in range(1,seq_len):
                additional_targets[f"image_{i}"] = "image"
            self.transform.add_targets(additional_targets)
        self.num_classes = num_classes
        
        # TODO: Make this better striding for test sets
        self.dict_of_strides = defaultdict(list)
        for g, v in self.df.groupby('video'):
            for i in range(0, v.fcount.unique()[0]-self.seq_len, self.seq_len):
                self.dict_of_strides[g].append(i)

        
        self.keys = []
        for k, v in self.dict_of_strides.items():
            for v_i in v:
                self.keys.append(f"{k}-{v_i}")
        
    def __len__(self):
        return len(self.keys)
    

    def get_video_selection(self, video_name, start, end):
        data = pickle.loads(self.txn.get(video_name.encode('ascii')))
        
        frames = []
        for i in range(start, end):
            frame = cv2.imdecode(
                np.frombuffer(data[i], np.uint8),
                flags=cv2.IMREAD_COLOR,
            )
            frame = cv2.cvtColor(frame, code=cv2.COLOR_BGR2RGB)
            frames.append(frame)

        return frames, np.arange(start, end)


    def prepare_item(self, video, start):
        frames, selection = self.get_video_selection(
            video, start, start+self.seq_len
        )
        
        df_vid = self.df[self.df.video==video]

        # df_select = df_vid[(df_vid.start_frame.isin(selection)) & (df_vid.end_frame.isin(selection))]
        
        b = [selection[0], selection[-1]]
        
        df_select = df_vid[df_vid[['start_frame','end_frame']].apply(prep_overlap(b),axis=1)]

        targets = torch.zeros(self.seq_len,self.num_classes+1)
        for index, row in df_select.iterrows():
            s = max(row['start_frame']-selection[0],0)
            duration = row['end_frame']-row['start_frame']+1

            e = min(row['start_frame']-selection[0]+duration,self.seq_len)
            targets[s:e,row['label']]=1.0
        targets[:,-1] = (targets.sum(axis=1)==0.0).float()
        
        logit_targets = (targets.sum(axis=0)>=1)[:-1].float()
        
        if self.transform:
            temp = {}
            
            for i, tmp in enumerate(frames):
                if i==0:
                    temp['image'] =tmp
                else:
                    temp[f'image_{i}'] = tmp
            temp = self.transform(**temp)
            frames = torch.stack([temp[f"image_{ik}"] if ik!=0 else temp['image'] for ik in range(self.seq_len) ])
        else:
            frames = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)


        return {
            "frames": frames.float(),
            "targets": targets,
            "logit_targets":logit_targets,
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

        item = self.keys[index]
        split_item = item.split('-')
        video = split_item[0]
        start = int(split_item[1])
        

        res = self.prepare_item(video, start)

        return {
            "index": torch.tensor(index),
            "video_name": video,
            "start_interval": start,
            "frames": res["frames"],
            "targets": res["targets"],
            "logit_targets": res["logit_targets"],
            # "item":item
        }


