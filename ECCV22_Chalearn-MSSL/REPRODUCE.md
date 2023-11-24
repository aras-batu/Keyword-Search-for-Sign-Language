

# Downloads


Download the pretrained models and preprocessed data: [download](https://drive.google.com/file/d/1FN0t3H5bAB6fL8lsjNPonSsr_fUanHR4/view?usp=sharing)

Extract to base repo directory


# Prepare docker

```bash
docker build -t chalearn .
docker run -it --gpus all -v <REPO_PATH>:/chalearn/ chalearn /bin/bash
cd /charlearn/
```

# (Optional) Prepare lmdb format of data

The test data lmdb data is provided in the above download `processed` folder, alternatively it can be recreated with the follow script:

```
python scripts/track1_lmdb_creator.py \
  --dest_dir processed/test/lmdb_videos/test \
  --dataset_dir <VIDEO_TEST_DIR> \
  --bbox_dir  chalearn_mssl/data/test_bboxes.json
```
This should be used also to create the train lmdb data and validation lmdb data replacing `test` with `train` and `valid` respectively.


# Inference

```bash
python main_results_creator_v2.py --lmdb_dir $(lmdb_dir) --checkpoint_dir $(checkpoint_dir) --save_dir $(save_dir)
```

Example:
```bash
python main_results_creator_v2.py --lmdb_dir chalearn_mssl/processed/test/lmdb_videos/test --checkpoint_dir chalearn_mssl/submissions/7fold_neg0/valinc_maxfix_neg0_fold0/ckpts/best_checkpoint_80_0.2793.pt --save_dir chalearn_results
```

Repeating for each checkpoint in the submissions folder. (NOTE: models trained with fold3 (`*_fold3`) were ignored during online test submissions)
This will create a `.pkl` file in the `save_dir` to be used for the ensemble.

Submssions trained with folder prefix `5fold_` were trained only on training dataset while `7fold_` were trained with a mixture of training and validation data.


# Ensemble

```python
python solution_test_creator_v2.py
```

Takes all the pickle files in `chalearn_results/test` and ensembles the results, saving in a `predictions.pkl` file.




# (Optional) Training

```
python main.py --config config/example_config.py
```

For the submission each of the folds (`original_split` [train data only] and `new_split` [train+validation data]) were trained 3 times each with a different `neg_prob` [0.0, 0.1, 0.5].
Therefore in the config file the following parameters were changed:
`csv_dir` (location of fold csv directory), `vol_path` (location of lmdb dataset directory), `neg_prob` (probability of selecting other parts of the video sequence)

NOTE: `chalearn_mssl/data/MSSL_TRAIN_SET_GT.pkl` needs to be added to the relevant directory provided by the challange dataset hosts.
