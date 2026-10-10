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
- `src/vesselwatch/detect/`     YOLO-OBB dataset export and training wrapper, two-stage pipeline with per-stage scoring
- `src/vesselwatch/monitor/`    CDSE search/download (`cdse.py`), tiling (`tiles.py`); inference, state and email not built
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
- `python -m vesselwatch.detect.yolo --config configs/detect/yolo11s_obb.yaml` exports the dataset, trains the detector, scores it once on test (needs the `detect` extra)
- `python -m vesselwatch.detect.pipeline --name <name> --detector <pt> --classifier-config <yaml> --classifier-weights <pt>` detect → crop → classify on the test split
- `python -m vesselwatch.detect.finland --name <name> --data-dir <folder> --detector NAME=<pt> --reference --classifier <yaml>=<pt>` scores detectors and tallies classifiers on the Finnish Sentinel-2 test scenes (needs `CDSE_USERNAME` / `CDSE_PASSWORD` for the first download)
- `python -m vesselwatch.inventory <models folder>` identifies saved weights/histories by content
- `python -m vesselwatch.verify_legacy --models-dir <models folder>` scores the nine original weights against the paper (`--shared-classes` rescored with the train class index)
- `python -m vesselwatch.monitor.run --aoi configs/aoi/<name>.geojson` (not built yet)
- `.venv/Scripts/python.exe -m pytest` (tests build their own tiny synthetic dataset)

Data and output folders come from `data_root` / `runs_dir` in the config, or the `VESSELWATCH_DATA` / `VESSELWATCH_RUNS` environment variables. `cache_dir` / `VESSELWATCH_CACHE` keeps extracted crops on disk.

## Rules
- Class index mapping comes from the train split and is saved with every checkpoint. Never rebuild it from val/test.
- Test split is touched once per experiment, after model selection on val. The dataset's own test split has no labels: the official val split is the test set, and validation is carved out of train by image (`val_fraction`, `split_seed`).
- Every run fixes a seed and writes config + metrics + class map to `runs/<id>/`.
- The monitor deploys Daniel's own models, even where a published model scores higher (decided 2026-10-09). Published models, such as the Finnish Environment Institute's Sentinel-2 YOLOv8 detector, are benchmarks: report both scores side by side and say the choice was deliberate.
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
Roadmap lives in `README.md`. Done: steps 3–10, 12, 13, 15 and 16. Next: batch 3, a short write-up of old vs. new results and a one-page Sentinel design note. The detector trains on the level 0 boxes (two classes) and only localises; the class comes from the crop classifier. Steps 11 and 14 are postponed. Step 1 done (inventory run on Drive 2026-10-03); step 2 is deferred until something local needs real data. Step 7 done (Colab, 2026-10-08): all nine original weights score within 0.01 points of the paper through the ported package; result in `5.Projects/models/legacy_verification.json`. `monitor/` does not exist yet.

The default setup is the corrected one (`configs/crops/`): `labels: vessel_crops` gives one sample per annotated vessel, cut along its oriented box, with the class list taken from the train file's categories. The original study, defects included, is kept behind two switches set in `configs/legacy/` (`labels: image_majority`, `per_split_classes: true`). All crop runs use Adam and differ only in model and learning rate; the model is selected on val accuracy.

Inventory result: all nine original runs are complete on Drive in `5.Projects/models`. The canonical files are `best_model_{Net|Net_Max|ResNet}_Level{1,2,3}.pth` and `{Net|Net_Max|ResNet}_history_Level{1,2,3}.json`; each history's best val accuracy matches the paper. Level 3 has 48 classes. Other files there are earlier level 1 experiments (`*_deeper`, `*_geometric_aug`, `ResNet101_finetune`, `vgg_custom`, unsuffixed `Net`); `final_model_Net_Max_deeper.pth` actually holds a small `Net`. `inventory.json` sits in the same folder.

Dataset facts (checked on Drive 2026-10-08): train 2,198 images / 10,860 annotations, val 550 / 3,103, test 687 images with no annotations, so a held-out test set has to come from the labelled splits. Each COCO annotation has an upright `bbox` and the four corners of the oriented box in `segmentation`; the VOC XML repeats the corners in `<polygon>` (its `<rotated_box>` is in local coordinates, do not use it) and adds `Img_Resolution` and `Ship_size`. Level 0 JSON files exist. Level 1's four classes include `Dock`.

Crop study results (Colab, 2026-10-08; test accuracy, macro-F1 in brackets), Net / Net_Max / ResNet101:
L1 85.1 (74.2) / 84.2 (68.2) / 90.2 (82.2), L2 59.5 (56.9) / 42.4 (26.0) / 68.4 (67.0), L3 58.9 (70.1) / 38.5 (16.1) / 70.0 (78.4). Majority baseline 38.0 at L1, 12.8 at L2 and L3. Per-run files are on Drive in `5.Projects/runs/crops_*`, summary in `study_summary.json`.

Detector and pipeline results (Colab, 2026-10-09, files in `5.Projects/runs/yolo11s_obb` and `pipeline_*.json`): detector on test P 88.8 / R 85.3 / mAP50 87.8 / mAP50-95 68.2 (val was about 5 points higher). Pipeline at conf 0.25, IoU 0.5: 87.4% of vessels found, 78.3% of detections match; correct class if found 91.6 / 68.9 / 71.4 at L1 / L2 / L3; found and correct 80.1 / 60.2 / 62.4. Level 2 comparisons: first prototype's setup 56.1 found, 19.9 correct if found, 11.2 end to end; stock yolo11n with the new classifier 65.5 / 60.3 / 39.5. The Colab cell output stopped displaying at epoch 96 but the run completed; whether it resumed or simply kept running is not known.

Open questions from that run:
- Net_Max trained unstably (val accuracy swinging between epochs, still underfitting at epoch 30). Its numbers reflect the recipe; a rerun with a lower learning rate or more epochs is undecided.
- Val ran 6 to 7 points above test at L2 and L3. Suspected cause: images are tiles of larger scenes (filenames like `1472__1840_0.bmp`), so neighbouring tiles land in both train and the carved-out val. Not verified; the fix would be to group the val split by source scene.
- Value of detector fine-tuning: `notebooks/colab_detect.ipynb` compares the fine-tuned `yolo11s` with the stock `yolo11n`, which mixes fine-tuning with model size. To separate them, also run the pipeline with the stock `yolo11s-obb.pt` (`--name stock_s_detector_level2 --keep-class ship`, new classifier). The stock models have no dock class, and about 5% of test objects are docks, which costs them a few points of recall.
- The Sentinel design note is for Sentinel-2 (optical), decided 2026-10-09; Sentinel-1 stays a Phase 5 extension.

Plan agreed 2026-10-09 (the Sentinel-2 experiment). Daniel does not want to label any data. Work proceeds one lettered step per batch.
- A. Run the current high-resolution detector and classifier on the Finnish Sentinel-2 test tiles, plus the published Finnish YOLOv8 model as the reference line. Expected: very few detections and meaningless types. Types cannot be scored there, so report the tally of predicted classes and how often a predicted class contradicts the measured size.
- B. Downgrade ShipRSImageNet to 10 m and retrain the detector and the classifier.
- C. Score both on the downgraded test set, which has boxes and types (favourable conditions).
- D. Run the 10 m pipeline on the Finnish test tiles again and compare with A. This is the key real-world number.
- E. Measurement logic (length, width, heading from the box; type plausibility). Later.
- F. Deploy the 10 m models on a live Sentinel-2 AOI with email alerts, with or without E. AOI not chosen; Baltic waters suit the Finnish benchmark and free Danish AIS.

The write-up is a separate track, independent of steps A to F and not waiting on any of them: a chronicle of the 2025 study, what was wrong and the rework. The "what I learned" sections are Daniel's to write. Venue and language not decided. Do not schedule it inside the experiment plan.

Where step A stands (2026-10-10): code and tests written (`detect/finland.py`, `detect/sizes.py`, `monitor/cdse.py`, `monitor/tiles.py`, `notebooks/colab_sentinel2.ipynb`), not yet run on Colab. Daniel has a Copernicus account; the login goes into Colab secrets (`CDSE_USERNAME`, `CDSE_PASSWORD`), never the repo. A local trial on a 12.8 km window of scene 34VEN 20220813, with L2A imagery from a free mirror: the Finnish reference found 29 of 45 vessels with no false alarms (all 13 with boxes over 100 m, 6 of 14 under 50 m); the stock DOTA detector found none. The real download path from Copernicus is untested until the notebook runs.

What the Finnish imagery looks like (inspected 2026-10-10): the labelled vessels are mostly small boats of a few bright pixels, often with a wake; the boxes include the wake, so even the largest boxes (150 to 280 m) are small fast boats, not large ships. The benchmark therefore measures small-boat and wake detection, which differs from both ShipRSImageNet (mostly moored ships, no wakes) and Daniel's eventual interest in large vessels. Report recall by box length. The IMT Atlantique/CLS set (60 scenes with image, land mask and per-ship CSV, downloaded but not yet inspected) may be the better match for large ships.

Finnish dataset facts (checked 2026-10-09): five GeoPackage files named by MGRS tile (34VEM, 34VEN, 34VER, 34WFT, 35VLG), one layer per acquisition date (`YYYYMMDD`), axis-aligned box polygons in the tile's UTM CRS (EPSG:32634, 32635 for 35VLG), single label `boat`. Test set is tiles 34VEN (20210714, 20220619, 20220624, 20220813; 1,339 boxes) and 34VER (20220617, 20220712, 20220826; 355 boxes). Imagery is the L1C true-colour image (TCI) from Copernicus; free no-login mirrors carry L2A, which would handicap the Finnish model. Their model takes 320 x 320 px chips upsampled to 640 x 640; use the same chips for every model. Their weights are AGPL-3.0: fine as a benchmark, not as part of the deployed monitor.

Public data for this, none of it needing labelling: Finnish coast Sentinel-2 vessel boxes (zenodo.org/records/15019034, 8,367 vessels, CC BY 4.0, annotations only, imagery from Copernicus by product name, weights at huggingface.co/mayrajeo/marine-vessel-yolo, licence of the weights unchecked); IMT Atlantique/CLS Sentinel-2 ships with length and heading (zenodo.org/records/10418786, 60 images, 1,147 ships, CC BY 4.0). No public Sentinel-2 set has ship types, so classification can only be scored on downgraded ShipRSImageNet. NAIP aerial imagery (Planetary Computer, free, 0.3 to 1 m) covers US naval bases and is safe to publish as example images.

`verify_legacy --shared-classes` showed the original level 3 scores were distorted by the per-split class index: with the train index the same weights score 12.73 / 10.73 / 50.73 instead of 9.64 / 10.36 / 23.09. Levels 1 and 2 do not change.

Historical results (val accuracy, image-level labels, 30 epochs), Net / Net_Max / ResNet101:
L1 (4 classes) 70 / 69 / 86, L2 (25) 19 / 18 / 51, L3 (48) 10 / 10 / 23.
