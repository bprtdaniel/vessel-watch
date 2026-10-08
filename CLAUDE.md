# vessel-watch

Vessel detection and classification from satellite imagery. Two tracks, one package.

- Track A: custom CNNs vs. pretrained ResNet vs. YOLO-OBB on ShipRSImageNet (0.12–6 m), at label levels 1/2/3.
- Track B: scheduled Sentinel-2 monitor over an AOI (Copernicus Data Space), emails on detections.

## Layout
- `src/vesselwatch/data/`       COCO parsing, vessel crops, datasets, transforms, splits
- `src/vesselwatch/models/`     small_cnn, vgg_style, resnet, registry (`build_model(name, num_classes)`)
- `src/vesselwatch/train.py`    single training loop, config-driven
- `src/vesselwatch/evaluate.py` scores weights on a split, writes metrics and confusion matrix
- `src/vesselwatch/metrics.py`  macro-F1, per-class, by-size, majority baseline
- `src/vesselwatch/study.py`    trains and scores every config in a folder, resumable
- `src/vesselwatch/detect/`     YOLO-OBB wrapper, rotated crop extraction
- `src/vesselwatch/monitor/`    CDSE search/download, tiling, inference, state, email
- `configs/`                    one YAML per experiment
- `notebooks/`                  thin Colab launchers only, no logic
- `legacy/`                     original Colab exports, read-only reference
- `docs/experiments/`           parked ideas, one file each; not roadmap work until Daniel says so
- `data/`, `runs/`              git-ignored

Most of this layout does not exist yet; see Status.

## Commands
- `python -m vesselwatch.train --config configs/crops/level1_resnet101.yaml`
- `python -m vesselwatch.evaluate --config configs/crops/level1_resnet101.yaml --weights <file> --split val|test`
- `python -m vesselwatch.study --configs configs/crops` runs all nine crop experiments, skipping finished ones
- `python -m vesselwatch.inventory <models folder>` identifies saved weights/histories by content
- `python -m vesselwatch.verify_legacy --models-dir <models folder>` scores the nine original weights against the paper (`--shared-classes` rescored with the train class index)
- `python -m vesselwatch.monitor.run --aoi configs/aoi/<name>.geojson` (not built yet)
- `.venv/Scripts/python.exe -m pytest` (tests build their own tiny synthetic dataset)

Data and output folders come from `data_root` / `runs_dir` in the config, or the `VESSELWATCH_DATA` / `VESSELWATCH_RUNS` environment variables. `cache_dir` / `VESSELWATCH_CACHE` keeps extracted crops on disk.

## Rules
- Class index mapping comes from the train split and is saved with every checkpoint. Never rebuild it from val/test.
- Test split is touched once per experiment, after model selection on val. The dataset's own test split has no labels: the official val split is the test set, and validation is carved out of train by image (`val_fraction`, `split_seed`).
- Every run fixes a seed and writes config + metrics + class map to `runs/<id>/`.
- The monitor deploys one model, chosen on the Track A evidence (expected: YOLO, since the monitor's job is detection). The custom CNNs and ResNet belong to the study and do not have to be deployed.
- Keep a thin `Predictor` interface so the deployed model can be swapped later.
- No secrets in the repo. CDSE and SMTP credentials come from environment variables / GitHub secrets.
- Do not claim fine-grained classes on Sentinel-2 output; 10 m supports detection and size class only.
- Training runs on Colab via `notebooks/colab_study.ipynb` (all runs) or `notebooks/colab_train.ipynb` (one run); the dataset and weights live on Google Drive (`5.Projects/Data/ShipRSImageNet_V1`, `5.Projects/models`), never in the repo. Locally only a small sample exists for tests and smoke runs.
- `legacy/` is reference only. Do not edit it; port code out of it.
- Work proceeds in batches of roadmap steps agreed with Daniel, stopping for a go-ahead after each batch and before each Colab session.

## Known defects in the legacy code (to fix while porting)
- One majority-vote label per multi-ship image (`legacy/level1.py:60`). Replace with per-vessel crops from ground-truth boxes.
- Class index built separately per split (`legacy/level1.py:121`).
- `ResNet.eval()` commented out in level 1 test inference (`legacy/level1.py:826`).
- Model selected and reported on the same val split; no held-out test; accuracy only.
- `RandomRotation(0.5)` is ±0.5 degrees; different optimiser per model; no seeds.

## Status
Roadmap lives in `README.md`. Done: steps 3–10, 12 and 13. Next: batch 2, steps 15 and 16 (YOLO-OBB detector with a single `ship` class, then detect → rotated crop → classify, each stage scored separately, with `Code/yolo11n-OBB_vessel_detection.ipynb` as the naive baseline), followed by a Colab session. After that: a short write-up of old vs. new results and a one-page Sentinel design note. Steps 11 and 14 are postponed. Step 1 done (inventory run on Drive 2026-10-03); step 2 is deferred until something local needs real data. Step 7 done (Colab, 2026-10-08): all nine original weights score within 0.01 points of the paper through the ported package; result in `5.Projects/models/legacy_verification.json`. `detect/` and `monitor/` do not exist yet.

The default setup is the corrected one (`configs/crops/`): `labels: vessel_crops` gives one sample per annotated vessel, cut along its oriented box, with the class list taken from the train file's categories. The original study, defects included, is kept behind two switches set in `configs/legacy/` (`labels: image_majority`, `per_split_classes: true`). All crop runs use Adam and differ only in model and learning rate; the model is selected on val accuracy.

Inventory result: all nine original runs are complete on Drive in `5.Projects/models`. The canonical files are `best_model_{Net|Net_Max|ResNet}_Level{1,2,3}.pth` and `{Net|Net_Max|ResNet}_history_Level{1,2,3}.json`; each history's best val accuracy matches the paper. Level 3 has 48 classes. Other files there are earlier level 1 experiments (`*_deeper`, `*_geometric_aug`, `ResNet101_finetune`, `vgg_custom`, unsuffixed `Net`); `final_model_Net_Max_deeper.pth` actually holds a small `Net`. `inventory.json` sits in the same folder.

Dataset facts (checked on Drive 2026-10-08): train 2,198 images / 10,860 annotations, val 550 / 3,103, test 687 images with no annotations, so a held-out test set has to come from the labelled splits. Each COCO annotation has an upright `bbox` and the four corners of the oriented box in `segmentation`; the VOC XML repeats the corners in `<polygon>` (its `<rotated_box>` is in local coordinates, do not use it) and adds `Img_Resolution` and `Ship_size`. Level 0 JSON files exist. Level 1's four classes include `Dock`.

Crop study results (Colab, 2026-10-08; test accuracy, macro-F1 in brackets), Net / Net_Max / ResNet101:
L1 85.1 (74.2) / 84.2 (68.2) / 90.2 (82.2), L2 59.5 (56.9) / 42.4 (26.0) / 68.4 (67.0), L3 58.9 (70.1) / 38.5 (16.1) / 70.0 (78.4). Majority baseline 38.0 at L1, 12.8 at L2 and L3. Per-run files are on Drive in `5.Projects/runs/crops_*`, summary in `study_summary.json`.

Open questions from that run:
- Net_Max trained unstably (val accuracy swinging between epochs, still underfitting at epoch 30). Its numbers reflect the recipe; a rerun with a lower learning rate or more epochs is undecided.
- Val ran 6 to 7 points above test at L2 and L3. Suspected cause: images are tiles of larger scenes (filenames like `1472__1840_0.bmp`), so neighbouring tiles land in both train and the carved-out val. Not verified; the fix would be to group the val split by source scene.
- Daniel has not said whether the Sentinel design note is for Sentinel-1 (SAR) or Sentinel-2 (optical); the repo assumes Sentinel-2.

`verify_legacy --shared-classes` showed the original level 3 scores were distorted by the per-split class index: with the train index the same weights score 12.73 / 10.73 / 50.73 instead of 9.64 / 10.36 / 23.09. Levels 1 and 2 do not change.

Historical results (val accuracy, image-level labels, 30 epochs), Net / Net_Max / ResNet101:
L1 (4 classes) 70 / 69 / 86, L2 (25) 19 / 18 / 51, L3 (48) 10 / 10 / 23.
