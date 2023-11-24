from tqdm import tqdm
from dataset.continuous_isolated_dataset import ContinuousIsolatedDataset
from dataset.continuous_dataset import ContinuousDataset
from metrics.classaccuracy import ClassAccuracy
from augmentation.augment import train_transform, valid_transform
import torch
from ignite.metrics import Average
import pickle

from PIL import Image

from metrics.solution_metric import SolutionMetric
from dataset.imbalanced_dataset_sampler import ImbalancedDatasetSampler


from augmentation.mixup import mixup_data, mixup_criterion
import ignite
import ignite.distributed as idist
from ignite.engine import Engine, Events
from ignite.contrib.handlers import ProgressBar
from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine

torch.version.cuda

import pprint


def training(cfg):  # local_rank, *args, **kwargs):
    # cfg = args[0]

    if cfg.logger_name == "wandb":
        import wandb
        from environment_variables import CONFIG
        import os

        for key, value in CONFIG.items():
            os.environ[key] = value

        writer = wandb.init(
            project="continuous_sign_spotting_task1",
            id=cfg.name,
            config=dict(cfg.hyperparams),
            save_code=False,
            tags=[],
            name=cfg.name,
            resume="allow",
            allow_val_change=True,
        )

    with open(cfg.gt_dir, "rb") as f:
        gt = pickle.load(f, encoding="bytes")

    train_ds = ContinuousIsolatedDataset(
        **cfg.train_ds_params, transform=train_transform
    )
    valid_ds = ContinuousDataset(**cfg.valid_ds_params, transform=valid_transform)

    train_dl = idist.auto_dataloader(
        train_ds,
        sampler=ImbalancedDatasetSampler(train_ds) if cfg.balance else None,
        shuffle=False if cfg.balance else True,
        num_workers=cfg.num_workers,
        batch_size=cfg.bs,
        drop_last=True,
        pin_memory=False,
    )
    valid_dl = idist.auto_dataloader(
        valid_ds,
        num_workers=cfg.num_workers,
        batch_size=cfg.bs,
        shuffle=False,
        drop_last=False,
        pin_memory=False,
    )

    if cfg.model_name == "upscalei3d":
        from models.upscale_i3d_detection import UpscaleI3dDectection

        model = UpscaleI3dDectection(**cfg.model_params)

    model = idist.auto_model(model)

    criterion_time = torch.nn.CrossEntropyLoss(label_smoothing=0.0)

    optimizer = torch.optim.AdamW(
        [{"params": model.parameters(), "lr": cfg.lr, "weight_decay": cfg.wd}]
    )
    optimizer = idist.auto_optim(optimizer)

    def train_step(engine, batch):
        engine.state.batch = None
        engine.state.output = None
        model.train()

        x = batch["frames"].cuda()
        targets = batch["targets"].cuda()

        b, s, c, h, w = x.shape
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(True):
            if cfg.mixup:
                mixed_x, y_a, y_b, lam = mixup_data(x, targets, alpha=1.0)

                y_pred = model(mixed_x)
                dict_losses = {}
                for x_key, x_pred in y_pred.items():
                    b_t, t_t, c_t = x_pred["time_out"].shape

                    with torch.no_grad():
                        in_y_a = torch.nn.functional.adaptive_max_pool1d(
                            y_a.permute(0, 2, 1), t_t
                        ).permute(0, 2, 1)

                        if cfg.maxfix:
                            in_y_a[:, :, -1] = (
                                in_y_a[:, :, :-1].sum(axis=-1) == 0.0
                            ).float()

                        in_y_b = torch.nn.functional.adaptive_max_pool1d(
                            y_b.permute(0, 2, 1), t_t
                        ).permute(0, 2, 1)

                        if cfg.maxfix:
                            in_y_b[:, :, -1] = (
                                in_y_b[:, :, :-1].sum(axis=-1) == 0.0
                            ).float()

                    time_loss = mixup_criterion(
                        criterion_time,
                        torch.nn.functional.adaptive_max_pool1d(
                            x_pred["time_out"].permute(0, 2, 1), t_t
                        )
                        .permute(0, 2, 1)
                        .reshape(-1, c_t),
                        in_y_a.reshape(-1, c_t),
                        in_y_b.reshape(-1, c_t),
                        lam,
                    )
                    dict_losses[x_key] = time_loss

                loss = 0
                for k, v in dict_losses.items():
                    loss += cfg.time_loss_dict[k] * v

            else:
                y_pred = model(x)
                dict_losses = {}
                for x_key, x_pred in y_pred.items():
                    b_t, t_t, c_t = x_pred["time_out"].shape
                    with torch.no_grad():
                        in_targets = torch.nn.functional.adaptive_max_pool1d(
                            targets.permute(0, 2, 1), t_t
                        ).permute(0, 2, 1)

                        if cfg.maxfix:
                            in_targets[:, :, -1] = (
                                in_targets[:, :, :-1].sum(axis=-1) == 0.0
                            ).float()
                    time_loss = criterion_time(
                        x_pred["time_out"].reshape(-1, c_t),
                        in_targets.reshape(-1, c_t),
                    )
                    dict_losses[x_key] = time_loss

                loss = 0
                for k, v in dict_losses.items():
                    loss += cfg.time_loss_dict[k] * v

        loss.backward()
        optimizer.step()

        return {
            "y_pred": y_pred,
            "batch": batch,
            "losses": {
                "loss": loss.detach(),
                **dict_losses,
            },
        }

    def eval_step(engine, batch):
        engine.state.batch = None
        engine.state.output = None
        model.eval()

        x = batch["frames"].cuda()
        targets = batch["targets"].cuda()

        b, s, c, h, w = x.shape
        with torch.no_grad():
            with torch.cuda.amp.autocast(True):

                y_pred = model(x)
                dict_losses = {}
                for x_key, x_pred in y_pred.items():
                    # y_pred = y_pred["x4"]

                    b_t, t_t, c_t = x_pred["time_out"].shape

                    with torch.no_grad():
                        in_targets = torch.nn.functional.adaptive_max_pool1d(
                            targets.permute(0, 2, 1), t_t
                        ).permute(0, 2, 1)
                        if cfg.maxfix:
                            in_targets[:, :, -1] = (
                                in_targets[:, :, :-1].sum(axis=-1) == 0.0
                            ).float()

                    time_loss = criterion_time(
                        x_pred["time_out"].reshape(-1, c_t),
                        in_targets.reshape(-1, c_t),
                    )
                    dict_losses[x_key] = time_loss
                loss = 0
                for k, v in dict_losses.items():
                    loss += cfg.time_loss_dict[k] * v

        return {
            "y_pred": y_pred,
            "batch": batch,
            "losses": {"loss": loss.detach(), **dict_losses},
        }

    trainer = Engine(train_step)
    evaluator = Engine(eval_step)

    # -------------------------------------- [START] LR SCHEDULER HANDLER --------------------------------------
    from ignite.handlers import CosineAnnealingScheduler, LRScheduler
    from torch.optim.lr_scheduler import MultiStepLR
    import math

    if cfg.lr_scheduler == "cosine":
        epoch_length = len(train_dl)
        num_cycles = 1
        cycle_size = math.ceil((cfg.max_epochs * epoch_length) / num_cycles)
        scheduler = CosineAnnealingScheduler(
            optimizer,
            "lr",
            start_value=cfg.lr,
            end_value=cfg.lr * 0.1,
            cycle_size=cycle_size,
        )
        trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    if cfg.lr_scheduler == "step":
        lr_scheduler = MultiStepLR(optimizer=optimizer, **cfg.lr_scheduler_params)
        scheduler = LRScheduler(lr_scheduler)
        trainer.add_event_handler(Events.EPOCH_STARTED, scheduler)

    # -------------------------------------- [END] LR SCHEDULER HANDLER --------------------------------------

    # -------------------------------------- [START] METRICS HANDLER --------------------------------------

    dict_train_metrics = {}
    dict_valid_metrics = {}

    dict_train_metrics["train_avg_loss"] = Average(
        output_transform=lambda x: x["losses"]["loss"]
    )
    dict_valid_metrics["valid_avg_loss"] = Average(
        output_transform=lambda x: x["losses"]["loss"]
    )

    def prep_loss_transform(ok):
        def get_loss(x):
            return x["losses"][ok]

        return get_loss

    def prep_acc_transform(ok):
        def get_acc(x):
            return x["y_pred"][ok]["time_out"], x["batch"]["targets"]

        return get_acc

    def prep_solution_transform(ok):
        def get_sol(x):
            return (x["y_pred"][ok], x["batch"])

        return get_sol

    for out_key in cfg.time_loss_dict.keys():
        dict_train_metrics[f"train_avg_{out_key}_loss"] = Average(
            output_transform=prep_loss_transform(out_key)
        )
        dict_valid_metrics[f"valid_avg_{out_key}_loss"] = Average(
            output_transform=prep_loss_transform(out_key)
        )

        dict_train_metrics[f"train_time_{out_key}_accuracy"] = ClassAccuracy(
            threshold=0.4,
            k=1,
            num_classes=cfg.num_classes + 1,
            output_transform=prep_acc_transform(out_key),
        )

        dict_valid_metrics[f"valid_time_{out_key}_accuracy"] = ClassAccuracy(
            threshold=0.4,
            k=1,
            num_classes=cfg.num_classes + 1,
            output_transform=prep_acc_transform(out_key),
        )

        dict_valid_metrics[f"valid_{out_key}_solution"] = SolutionMetric(
            gt=gt,
            num_classes=cfg.num_classes,
            output_transform=prep_solution_transform(out_key),
        )

    for k, v in dict_train_metrics.items():
        v.attach(trainer, k)

    for k, v in dict_valid_metrics.items():
        v.attach(evaluator, k)

    # -------------------------------------- [END] METRICS HANDLER --------------------------------------

    # -------------------------------------- [START] CHECKPOINT HANDLER --------------------------------------

    def score_function(engine):
        return float(engine.state.metrics["valid_x_solution"]["f1"])

    checkpoint = Checkpoint(
        {
            "trainer": trainer,
            "optimizer": optimizer,
            "model": model,
            "scheduler": scheduler,
        },
        DiskSaver(dirname=cfg.save_dir, require_empty=False),
        n_saved=3,
        filename_prefix=f"best",
        score_function=score_function,
        score_name=None,
        global_step_transform=global_step_from_engine(trainer),
        greater_or_equal=True,
    )

    latest_checkpoint = Checkpoint(
        {
            "trainer": trainer,
            "optimizer": optimizer,
            "model": model,
            "scheduler": scheduler,
        },
        DiskSaver(dirname=cfg.save_dir, require_empty=False),
        n_saved=1,
        filename_prefix=f"latest",
    )

    def run_evaluator(engine):
        evaluator.run(valid_dl, max_epochs=1, epoch_length=None)

    evaluator.add_event_handler(Events.EPOCH_COMPLETED(every=1), checkpoint)
    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=1), latest_checkpoint)

    # -------------------------------------- [END] CHECKPOINT HANDLER --------------------------------------

    # -------------------------------------- [START] LOADING / RESUME HANDLER --------------------------------------
    if cfg.model_checkpoint_dir != "":
        print("LOADING MODEL FROM:", cfg.model_checkpoint_dir)
        objects_to_load = {"model": model}
        Checkpoint.load_objects(
            to_load=objects_to_load, checkpoint=cfg.model_checkpoint_dir
        )

    if cfg.resume:
        from configs.checkpoint_helpers import get_latest_saved_file

        latest_checkpoint_from_resume = get_latest_saved_file(
            cfg.save_dir, extension="pt"
        )
        if latest_checkpoint_from_resume[0]:
            objects_to_load = {
                "model": model,
                "optimizer": optimizer,
                "trainer": trainer,
                "scheduler": scheduler,
            }

            Checkpoint.load_objects(
                to_load=objects_to_load, checkpoint=latest_checkpoint_from_resume[0]
            )

    # -------------------------------------- [END] LOADING / RESUME HANDLER --------------------------------------

    # -------------------------------------- [START] LOGGING HANDLER --------------------------------------

    @trainer.on(Events.EPOCH_COMPLETED(every=1))
    def print_output(engine):
        if cfg.logger_name == "wandb":
            for k, v in engine.state.metrics.items():
                if type(v) == dict:
                    for k_i, v_i in v.items():
                        writer.log({f"k_{k_i}": v_i, f"epoch": engine.state.epoch})
                else:
                    writer.log({f"{k}": v, f"epoch": engine.state.epoch})
        print("TRAINER", engine.state.epoch)
        pprint.pprint(engine.state.metrics)

    @trainer.on(Events.EPOCH_COMPLETED(every=1))
    def print_lr(engine):

        for i, pg in enumerate(optimizer.param_groups):
            pprint.pprint({f"train/lr_{i}": optimizer.param_groups[i]["lr"]})
            if cfg.logger_name == "wandb":
                writer.log(
                    {
                        f"train/lr_{i}": optimizer.param_groups[i]["lr"],
                        f"global_step": engine.state.epoch,
                    }
                )

    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=5), run_evaluator)

    @evaluator.on(Events.EPOCH_COMPLETED(every=1))
    def print_output(engine):
        if cfg.logger_name == "wandb":
            for k, v in engine.state.metrics.items():
                if type(v) == dict:
                    for k_i, v_i in v.items():
                        writer.log({f"{k}_{k_i}": v_i, f"epoch": trainer.state.epoch})
                else:
                    writer.log({f"{k}": v, f"epoch": trainer.state.epoch})
        print("VALID: ", trainer.state.epoch)
        pprint.pprint(engine.state.metrics)

    pbar = ProgressBar()
    pbar.attach(evaluator)

    pbar = ProgressBar()
    pbar.attach(trainer)

    # -------------------------------------- [END] LOGGING HANDLER --------------------------------------

    trainer.run(train_dl, max_epochs=cfg.max_epochs, epoch_length=None)


from ml_collections import config_flags
from absl import app

CONFIG = config_flags.DEFINE_config_file(
    "config",
    default="configs/testing/config.py",
)


def main(_):
    # backend = "nccl"  # or 'gloo', 'horovod', 'xla-tpu'
    # with idist.Parallel(backend) as parallel:
    #     parallel.run(training, CONFIG.value)
    training(CONFIG.value)


if __name__ == "__main__":

    app.run(main)
