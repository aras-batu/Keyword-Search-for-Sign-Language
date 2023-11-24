from ml_collections import config_dict
from pathlib import Path
import os


def get_config():
    vol_path = "train/lmdb_videos/train"

    cfg = config_dict.ConfigDict()
    cfg.model_name = "upscalei3d"
    cfg.fold = 0
    cfg.num_classes = 60
    cfg.seq_len = 32
    cfg.balance = True
    cfg.maxfix = True
    cfg.lr_scheduler = "cosine"
    cfg.lr_scheduler_params = {}  # {"milestones": [100, 150], "gamma": 0.1}

    cfg.lr = 3e-4
    cfg.wd = 0.0
    cfg.mixup = True

    cfg.max_epochs = 200

    cfg.time_loss_dict = {
        "x4": 1 / 5,
        "x8": 1 / 5,
        "x16": 1 / 5,
        "x32": 1 / 5,
        "x": 1 / 5,
    }

    cfg.bs = 8
    cfg.num_workers = 8

    cfg.model_checkpoint_dir = ""

    train_ds_params = {
        "lmdb_dir": f"{vol_path}",
        "csv_dir": f"chalearn_mssl/data/original_split/train_{cfg.fold}.csv",
        "seq_len": cfg.seq_len,
        "num_classes": cfg.num_classes,
        "neg_prob": 0.0,
    }
    cfg.train_ds_params = config_dict.ConfigDict(train_ds_params)

    valid_ds_params = {
        "lmdb_dir": f"{vol_path}",
        "csv_dir": f"chalearn_mssl/data/original_split/valid_{cfg.fold}.csv",
        "seq_len": cfg.seq_len,
        "num_classes": cfg.num_classes,
    }
    cfg.valid_ds_params = config_dict.ConfigDict(valid_ds_params)

    i3d_model_params = {
        "num_classes": cfg.num_classes,
        "ckpt_path": "chalearn_mssl/data/pretrained_models/model_bsl1k_wlasl.pth.tar",
        "activation": "swish",
    }

    head_model_params = {
        "num_classes": cfg.num_classes,
        "input_dim": 1024,
        "dropout_seg": 0.5,
        "dropout_time": 0.5,
    }

    model_params = {
        "i3d_model_params": i3d_model_params,
        "head_model_params": head_model_params,
    }

    cfg.model_params = config_dict.ConfigDict(model_params)

    cfg.save_dir = f"{Path(__file__).parent.resolve()}/ckpts"
    cfg.gt_dir = "chalearn_mssl/data/MSSL_TRAIN_SET_GT.pkl"

    cfg.resume = True

    cfg.name = (os.path.dirname(os.path.realpath(__file__))).split("/")[-1]
    cfg.hyperparams = {
        "model": cfg.model_name,
        "fold": cfg.fold,
        "seq_len": cfg.seq_len,
        "balance": cfg.balance,
        "maxfix": cfg.maxfix,
        "lr": cfg.lr,
        "wd": cfg.wd,
        "mixup": cfg.mixup,
        "max_epochs": cfg.max_epochs,
        **cfg.time_loss_dict,
        "bs": cfg.bs,
        "lr_scheduler": cfg.lr_scheduler,
        **cfg.lr_scheduler_params,
    }
    cfg.logger_name = "text"
    print(cfg)
    return cfg
