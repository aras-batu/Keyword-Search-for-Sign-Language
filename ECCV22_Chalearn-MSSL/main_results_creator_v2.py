from tqdm.auto import tqdm
from augmentation.augment import valid_transform
import torch
from metrics.ineval import compute_f1
import pickle
from pathlib import Path

from PIL import Image
from metrics.utils.solution_utils import (
    get_solution,
    beam_search_decoder,
    compute_intervals,
)
from models.upscale_i3d_detection import UpscaleI3dDectection

from reader_utils.reader import get_frames, transform_frames, get_data_point
from argparse import ArgumentParser

parser = ArgumentParser(parents=[])


parser.add_argument(
    "--lmdb_dir",
    type=str,
    default="lmdb_videos/test", 
)

parser.add_argument(
    "--checkpoint_dir",
    type=str,
    default="",
)

parser.add_argument(
    "--save_dir",
    type=str,
    default="", 
)


params, unknown = parser.parse_known_args()


lmdb_dir = params.lmdb_dir
checkpoint_dir = params.checkpoint_dir
save_dir = params.save_dir

split = lmdb_dir.split("/")[-1]

save_folder = f"{checkpoint_dir.split('/')[-3]}_{checkpoint_dir.split('.')[-2]}"
Path(f"{save_dir}/{split}/{save_folder}").mkdir(parents=True, exist_ok=True)
seq_len = 32
num_classes = 60

i3d_model_params = {
    "num_classes": num_classes,
    "ckpt_path": None,
    "activation": "swish",
}

head_model_params = {
    "num_classes": num_classes,
    "input_dim": 1024,
    "dropout_seg": 0.5,
    "dropout_time": 0.5,
}

model_params = {
    "i3d_model_params": i3d_model_params,
    "head_model_params": head_model_params,
}


model = UpscaleI3dDectection(i3d_model_params, head_model_params)
ckpt = torch.load(checkpoint_dir)
r = model.load_state_dict(ckpt["model"])
print(r)

model.cuda()
model.eval()

data = get_data_point(lmdb_dir, "details")
list_of_item_names = list(data.keys())

dict_results = {}
for item_name in tqdm(list_of_item_names):
    item_data = get_data_point(lmdb_dir, item_name)
    item_frames = get_frames(item_data)
    norm_frames = transform_frames(item_frames, valid_transform)

    with torch.no_grad():
        list_of_time_outs_x = []
        list_of_time_outs_x4 = []
        list_of_time_outs_x8 = []
        list_of_time_outs_x16 = []
        list_of_time_outs_x32 = []
        list_of_x = []
        for i in tqdm(range(0, norm_frames.shape[0] - seq_len), leave=False):
            x = norm_frames[i : i + seq_len].cuda()
            list_of_x.append(x)
            if len(list_of_x) >= 8:
                x = torch.stack(list_of_x)
                y_pred = model(x)
                list_of_time_outs_x.append(y_pred["x"]["time_out"].detach().cpu())
                list_of_time_outs_x4.append(y_pred["x4"]["time_out"].detach().cpu())
                list_of_time_outs_x8.append(y_pred["x8"]["time_out"].detach().cpu())
                list_of_time_outs_x16.append(y_pred["x16"]["time_out"].detach().cpu())
                list_of_time_outs_x32.append(y_pred["x32"]["time_out"].detach().cpu())
                list_of_x = []
        list_of_time_outs_x = torch.vstack(list_of_time_outs_x)
        list_of_time_outs_x4 = torch.vstack(list_of_time_outs_x4)
        list_of_time_outs_x8 = torch.vstack(list_of_time_outs_x8)
        list_of_time_outs_x16 = torch.vstack(list_of_time_outs_x16)
        list_of_time_outs_x32 = torch.vstack(list_of_time_outs_x32)
        dict_results[item_name] = {
            "x": list_of_time_outs_x,
            "x4": list_of_time_outs_x4,
            "x8": list_of_time_outs_x8,
            "x16": list_of_time_outs_x16,
            "x32": list_of_time_outs_x32
        }


with open(f"{save_dir}/{split}/{save_folder}/dict_results.pkl", "wb") as handle:
    pickle.dump(dict_results, handle, protocol=4)
