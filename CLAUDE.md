# vessel-watch

Vessel detection and classification from satellite imagery. Two tracks, one package.

- Track A: custom CNNs vs. pretrained ResNet vs. YOLO-OBB on ShipRSImageNet (0.12–6 m), at label levels 1/2/3.
- Track B: scheduled Sentinel-2 monitor over an AOI (Copernicus Data Space), emails on detections.

## Layout
- `src/vesselwatch/data/`       COCO parsing, crop dataset, transforms, splits
- `src/vesselwatch/models/`     small_cnn, vgg_style, resnet, registry (`build_model(name, num_classes)`)
- `src/vesselwatch/train.py`    single training loop, config-driven
- `src/vesselwatch/evaluate.py` metrics, confusion matrix, report figures
- `src/vesselwatch/detect/`     YOLO-OBB wrapper, rotated crop extraction
- `src/vesselwatch/monitor/`    CDSE search/download, tiling, inference, state, email
- `configs/`                    one YAML per experiment
- `notebooks/`                  thin Colab launchers only, no logic
- `legacy/`                     original Colab exports, read-only reference
- `data/`, `runs/`              git-ignored

Most of this layout does not exist yet; see Status.

## Commands
- `python -m vesselwatch.train --config configs/legacy/level1_resnet101.yaml`
- `python -m vesselwatch.evaluate --config configs/legacy/level1_resnet101.yaml --weights <file>`
- `python -m vesselwatch.monitor.run --aoi configs/aoi/<name>.geojson` (not built yet)
- `.venv/Scripts/python.exe -m pytest` (tests build their own tiny synthetic dataset)

Data and output folders come from `data_root` / `runs_dir` in the config, or the `VESSELWATCH_DATA` / `VESSELWATCH_RUNS` environment variables.

## Rules
- Class index mapping comes from the train split and is saved with every checkpoint. Never rebuild it from val/test.
- Test split is touched once per experiment, after model selection on val.
- Every run fixes a seed and writes config + metrics + class map to `runs/<id>/`.
- All detectors/classifiers implement the `Predictor` interface so the monitor can swap backends.
- No secrets in the repo. CDSE and SMTP credentials come from environment variables / GitHub secrets.
- Do not claim fine-grained classes on Sentinel-2 output; 10 m supports detection and size class only.
- Training runs on Colab via `notebooks/colab_train.ipynb`; the dataset and weights live on Google Drive (`5.Projects/Data/ShipRSImageNet_V1`, `5.Projects/models`), never in the repo. Locally only a small sample exists for tests and smoke runs.
- `legacy/` is reference only. Do not edit it; port code out of it.
- Work proceeds one numbered roadmap step at a time, stopping for a go-ahead after each.

## Known defects in the legacy code (to fix while porting)
- One majority-vote label per multi-ship image (`legacy/level1.py:60`). Replace with per-vessel crops from ground-truth boxes.
- Class index built separately per split (`legacy/level1.py:121`).
- `ResNet.eval()` commented out in level 1 test inference (`legacy/level1.py:826`).
- Model selected and reported on the same val split; no held-out test; accuracy only.
- `RandomRotation(0.5)` is ±0.5 degrees; different optimiser per model; no seeds.

## Status
Roadmap lives in `README.md`. Done: steps 3–6. `data/`, `models/`, `train.py`, `evaluate.py`, `configs/legacy/` and the Colab launcher exist; `detect/` and `monitor/` do not.

The package currently reproduces the original study, defects included, behind two config switches (`labels: image_majority`, `per_split_classes: true`) so step 7 can check it against the saved weights. Phase 2 changes the defaults.

Historical results (val accuracy, image-level labels, 30 epochs), Net / Net_Max / ResNet101:
L1 (4 classes) 70 / 69 / 86, L2 (25) 19 / 18 / 51, L3 (48) 10 / 10 / 23.
