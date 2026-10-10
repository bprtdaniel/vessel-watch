# Notebooks: what was run on Colab, in order

Everything that runs on Google Colab is kept here, one folder per step of the project, in the order the steps happened. The notebooks are launchers: each mounts Google Drive, fetches this repository, and calls one command from the `vesselwatch` package. The logic is in `src/vesselwatch/`, and each step below says which file does the work.

Data, trained weights and results are on Google Drive, not in this repository:

- dataset: `5.Projects/Data/ShipRSImageNet_V1`
- weights of the 2025 study: `5.Projects/models`
- runs and results of the reworked study: `5.Projects/runs`
- Sentinel-2 scenes: `5.Projects/Data/sentinel2_finland`

To run a notebook, open it in Colab from GitHub (File > Open notebook > GitHub > `bprtdaniel/vessel-watch`) and choose Runtime > Run all.

## Step 1: inventory of the 2025 models

| | |
|---|---|
| Notebook | [`step_01_inventory/01_inventory_saved_models.ipynb`](step_01_inventory/01_inventory_saved_models.ipynb) |
| Question | Which saved weights and training histories from the 2025 study exist, and which belong to which model and label level? |
| Runs | `python -m vesselwatch.inventory <models folder>` |
| Code | [`src/vesselwatch/inventory.py`](../src/vesselwatch/inventory.py) |
| Writes | `inventory.json` in `5.Projects/models` |
| Run on | 2026-10-03 |
| Result | All nine original runs (three models at three levels) are complete. |

## Step 2: does the package reproduce the 2025 study?

| | |
|---|---|
| Notebook | [`step_02_verify_original_study/02_verify_original_study.ipynb`](step_02_verify_original_study/02_verify_original_study.ipynb) |
| Question | Does the rewritten code give the same scores as the original notebooks for the same weights? |
| Runs | `python -m vesselwatch.verify_legacy --models-dir <models folder>` |
| Code | [`src/vesselwatch/verify_legacy.py`](../src/vesselwatch/verify_legacy.py), [`evaluate.py`](../src/vesselwatch/evaluate.py), configs in [`configs/legacy/`](../configs/legacy/) |
| Writes | `legacy_verification.json` in `5.Projects/models` |
| Run on | 2026-10-08 |
| Result | All nine scores match the paper to two decimals. |

## Step 3: what labels does the dataset have?

| | |
|---|---|
| Notebook | [`step_03_check_dataset/03_check_dataset_labels.ipynb`](step_03_check_dataset/03_check_dataset_labels.ipynb) |
| Question | Does the test split have labels, and are oriented boxes stored? |
| Runs | A few lines that read the annotation files; no package code |
| Writes | Nothing |
| Run on | 2026-10-08 |
| Result | The test split has no labels, so the official val split became the held-out test set. Oriented boxes are in every annotation. |

## Step 4: the classifier study on single-vessel crops

| | |
|---|---|
| Notebooks | [`04a_train_all_classifiers.ipynb`](step_04_classifier_study/04a_train_all_classifiers.ipynb): the study. [`04b_train_single_classifier.ipynb`](step_04_classifier_study/04b_train_single_classifier.ipynb): one run only, for trying a config. [`04c_show_per_class_results.ipynb`](step_04_classifier_study/04c_show_per_class_results.ipynb): prints results per class and per vessel size. |
| Question | How do a small CNN, a VGG-style CNN and a fine-tuned ResNet101 compare when each vessel is cut out and classified on its own? Also: did the original class numbering distort the 2025 scores? |
| Runs | `python -m vesselwatch.verify_legacy --shared-classes`, then `python -m vesselwatch.study --configs configs/crops` |
| Code | [`study.py`](../src/vesselwatch/study.py) calls [`train.py`](../src/vesselwatch/train.py) and [`evaluate.py`](../src/vesselwatch/evaluate.py). Crops: [`data/crops.py`](../src/vesselwatch/data/crops.py), [`data/datasets.py`](../src/vesselwatch/data/datasets.py). Metrics: [`metrics.py`](../src/vesselwatch/metrics.py). Configs: [`configs/crops/`](../configs/crops/) |
| Writes | One folder per run, `crops_level{1,2,3}_{net,net_max,resnet101}`, in `5.Projects/runs`, with weights, training history, `metrics_val.json`, `metrics_test.json` and confusion matrices; `study_summary.json`; `legacy_verification_shared_classes.json` in `5.Projects/models` |
| Run on | 2026-10-08 (4a), 2026-10-09 (4c) |
| Result | ResNet101 reaches 90.2 / 68.4 / 70.0% test accuracy at 4 / 25 / 50 classes. The 2025 level 3 score of 23% was an evaluation error; the same weights score 51% with consistent class numbering. |

## Step 5: the detector and the two-stage pipeline

| | |
|---|---|
| Notebooks | [`05a_train_detector_and_run_pipeline.ipynb`](step_05_detector_and_pipeline/05a_train_detector_and_run_pipeline.ipynb): the experiment. [`05b_show_results.ipynb`](step_05_detector_and_pipeline/05b_show_results.ipynb): prints training progress and all results. [`05c_stock_small_detector_comparison.ipynb`](step_05_detector_and_pipeline/05c_stock_small_detector_comparison.ipynb): an extra comparison, **not run yet**. |
| Question | Can a detector find the vessels, and how good is detect, cut out, classify from end to end, compared with the first prototype? |
| Runs | `python -m vesselwatch.detect.yolo --config configs/detect/yolo11s_obb.yaml`, then `python -m vesselwatch.detect.pipeline` five times: levels 1, 2 and 3, the prototype's setup, and the stock detector with the new classifier |
| Code | [`detect/export.py`](../src/vesselwatch/detect/export.py) writes the dataset in YOLO format, [`detect/yolo.py`](../src/vesselwatch/detect/yolo.py) trains and scores the detector, [`detect/pipeline.py`](../src/vesselwatch/detect/pipeline.py) runs and scores the two stages |
| Writes | `yolo11s_obb/` (weights, training log, `metrics_test.json`) and `pipeline_<name>.json` files in `5.Projects/runs` |
| Run on | 2026-10-09 |
| Result | The detector finds 87.4% of vessels. Found and correctly classified: 80.1 / 60.2 / 62.4% at levels 1 / 2 / 3, against 11.2% at level 2 for the prototype's setup. |

## Step 6: the high-resolution models on real Sentinel-2 imagery

This is step A of the Sentinel-2 experiment.

| | |
|---|---|
| Notebook | [`step_06_sentinel2_current_models/06_current_models_on_sentinel2.ipynb`](step_06_sentinel2_current_models/06_current_models_on_sentinel2.ipynb) |
| Question | Do the models trained on 0.12 to 6 m imagery work on 10 m Sentinel-2 imagery? |
| Needs | A Copernicus Data Space login stored as the Colab secrets `CDSE_USERNAME` and `CDSE_PASSWORD` |
| Runs | `python -m vesselwatch.detect.finland --name step_a ...` |
| Code | [`detect/finland.py`](../src/vesselwatch/detect/finland.py) reads the Finnish vessel labels, runs the detectors chip by chip, scores them and tallies the classifiers. [`monitor/cdse.py`](../src/vesselwatch/monitor/cdse.py) downloads the scenes from Copernicus, [`monitor/tiles.py`](../src/vesselwatch/monitor/tiles.py) cuts them into chips, [`detect/sizes.py`](../src/vesselwatch/detect/sizes.py) holds the ship-class lengths |
| Writes | Scenes and labels to `5.Projects/Data/sentinel2_finland`; `sentinel2_step_a.json` and `sentinel2_step_a_examples/` to `5.Projects/runs` |
| Run on | 2026-10-10 |
| Result | The high-resolution detector found 5 of 1,694 labelled vessels (0.3%). The published Finnish model, trained on real Sentinel-2, found 90.1%. The classifiers gave nearly the same answer for every vessel. |

## Step 7: resampling the dataset to 10 m and retraining

Steps B and C of the Sentinel-2 experiment: two measurements (7a, 7b) that had to come first, then the retraining (7c).

| | |
|---|---|
| Notebook | [`step_07_downgrade_to_10m/07a_measure_resolution_and_vessel_sizes.ipynb`](step_07_downgrade_to_10m/07a_measure_resolution_and_vessel_sizes.ipynb) |
| Question | Which resolution does each image have, how small do the scenes become at 10 m, and how many vessels are still more than a pixel or two? |
| Runs | `python -m vesselwatch.data.resolution --target 10` |
| Code | [`data/resolution.py`](../src/vesselwatch/data/resolution.py) |
| Writes | Nothing |
| Run on | 2026-10-10 |
| Result | Only 1,257 of the 2,748 labelled images have a recorded resolution (0.3, 0.92, 1.07 or 4 m), covering 7,134 of 13,963 vessels. The lengths that follow from the recorded values are too large to be true (median warship 355 m), so the recorded values cannot be used as they are. |

| | |
|---|---|
| Notebook | [`step_07_downgrade_to_10m/07b_check_resolution_by_source.ipynb`](step_07_downgrade_to_10m/07b_check_resolution_by_source.ipynb) |
| Question | For each group of images with the same origin: what resolution is recorded, and what resolution do the images really have, measured from vessels whose hull length is known? |
| Runs | `python -m vesselwatch.data.resolution --sources` |
| Code | [`data/resolution.py`](../src/vesselwatch/data/resolution.py), `source_report`; hull lengths in [`detect/sizes.py`](../src/vesselwatch/detect/sizes.py) |
| Writes | Nothing |
| Run on | 2026-10-10 |
| Result | The dataset has six origins. FGSD (Google Earth, 1,470 images, 6,808 vessels) records no resolution and measures 0.54 m per pixel (0.25 to 0.60). HRSC (Google Earth, 819 images, 2,720 vessels) records 1.07 m but measures 0.49 m (0.42 to 0.56). xView (WorldView-3, 421 images, 4,397 vessels) records 0.3 m and has no named ship classes to check it against. The rest is 38 images: JL-1 (0.92 m), GF-2 (4 m) and Airbus (no value). |

| | |
|---|---|
| Notebook | [`step_07_downgrade_to_10m/07c_retrain_at_10m.ipynb`](step_07_downgrade_to_10m/07c_retrain_at_10m.ipynb) |
| Question | How well can vessels be detected and classified at 10 m, when the models are trained on ShipRSImageNet resampled to that resolution? |
| Runs | `python -m vesselwatch.study --configs configs/crops_10m`, then `python -m vesselwatch.detect.yolo --config configs/detect/yolo11s_obb_10m.yaml` |
| Code | Resolution per image: `assigned_resolutions` in [`data/resolution.py`](../src/vesselwatch/data/resolution.py). Shrinking images and boxes: [`data/resample.py`](../src/vesselwatch/data/resample.py). Fixed-scale crops: `fixed_crop` in [`data/crops.py`](../src/vesselwatch/data/crops.py). Canvases for the detector: [`detect/canvases.py`](../src/vesselwatch/detect/canvases.py). Training and scoring are the same code as steps 4 and 5. Configs: [`configs/crops_10m/`](../configs/crops_10m/), [`configs/detect/yolo11s_obb_10m.yaml`](../configs/detect/yolo11s_obb_10m.yaml) |
| Writes | `crops10m_level{1,2,3}_{net,net_max,resnet101}/`, `study_summary_10m.json` and `yolo11s_obb_10m/` in `5.Projects/runs` |
| Run on | not run yet |
| Result | - |

## Step 8: the 10 m models on real Sentinel-2 imagery

This is step D of the Sentinel-2 experiment.

| | |
|---|---|
| Notebook | [`step_08_sentinel2_10m_models/08_10m_models_on_sentinel2.ipynb`](step_08_sentinel2_10m_models/08_10m_models_on_sentinel2.ipynb) |
| Question | How much better do the models retrained at 10 m do on real Sentinel-2 scenes than the high-resolution models of step 6, and how far are they from a model trained on real Sentinel-2? |
| Needs | The runs of step 7c on Drive |
| Runs | `python -m vesselwatch.detect.finland --name step_d ...` with three detectors: retrained at 10 m, high-resolution, Finnish reference |
| Code | [`detect/finland.py`](../src/vesselwatch/detect/finland.py), as in step 6 |
| Writes | `sentinel2_step_d.json` and `sentinel2_step_d_examples/` in `5.Projects/runs` |
| Run on | not run yet |
| Result | - |

## Adding a step

A new Colab run gets its own folder, `step_NN_short_name/`, and a notebook named `NN_what_it_does.ipynb`, with a letter after the number where one step has several notebooks. Its first cell says what the step is for. One-off cells that were only pasted into Colab are saved here as well, so that every result can be traced back to the code that produced it. Add the step to this page with its date and result once it has run.
