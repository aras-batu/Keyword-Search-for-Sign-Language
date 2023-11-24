from pathlib import Path

import glob


def get_best_checkpoint_details(path):
    list_of_checkpoints = list(Path(path).glob("*.pt"))
    max_score = -100
    for pt in list_of_checkpoints:
        path_cpkt = str(pt).split("_checkpoint_")
        path_cpkt = path_cpkt[1].split("_")
        epoch = path_cpkt[0]
        score = float(path_cpkt[1].split(".pt")[0])
        if max_score < score:
            best_checkpoint = pt
            best_epoch = epoch
            best_score = score
            max_score = score

    print(best_checkpoint, best_epoch, best_score)
    return str(best_checkpoint), best_epoch, best_score


def get_latest_saved_file(folder, extension="pt"):
    list_of_files = list(glob.glob(f"{folder}/*.{extension}"))
    latest_checkpoint = None
    highest_num = 0
    for f in list_of_files:
        if "latest" in Path(f).stem:
            num = int(f[f.find("latest_checkpoint") :].split('_')[-1].replace(f'.{extension}', ''))
            if num >= highest_num:
                latest_checkpoint = f
                highest_num = num
    print(latest_checkpoint)
    return latest_checkpoint, -1, -1


def get_epoch_saved_file(folder, epoch, extension="pt"):
    list_of_files = list(glob.glob(f"{folder}/*.{extension}"))
    latest_checkpoint = None
    highest_num = 0
    for f in list_of_files:
        num = int(f[f.find("checkpoint_") :].split("_")[1])
        if num == epoch:
            latest_checkpoint = f
            highest_num = num
    print(latest_checkpoint)
    return latest_checkpoint, -1, -1
