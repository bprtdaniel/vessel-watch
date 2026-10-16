# Pathway: from the 2025 study to a Sentinel-2 detector

The record of every stage of the project: what each asked, how it was set up, what came out, and why the next stage followed. It is the source for the write-up, not the write-up itself.

- Each stage lists its question, what triggered it, the settings, the result, the decision that followed, and where the evidence is.
- **Gap** marks something not yet on record. The open gaps are collected at the end.
- Result files are on Google Drive under `5.Projects/` (`models/` for the 2025 weights, `runs/` for everything since). Step 10b gathers them all into `docs/results/pathway_results.json`.
- Notebook steps (`step_NN`) are the Colab runs in [`notebooks/`](../notebooks/README.md). Letters A to F are the stages of the Sentinel-2 plan agreed on 2026-10-09.

## Overview

| # | Stage | Question | Answer | What it led to |
|---|---|---|---|---|
| 0 | 2025 study | Small CNNs vs. pretrained ResNet101 at 4 / 25 / 48 classes | ResNet best: 86 / 51 / 23% val accuracy; pipeline "disappointing" | Audit of the method |
| 1 | Audit and port | Does the rewritten code reproduce 2025, and what was wrong? | Reproduced to 0.01 points; five defects found | Fix the method |
| 2 | Corrected classifier study | What do the models score with one sample per vessel and a held-out test set? | ResNet101 90.2 / 68.4 / 70.0% test accuracy | Build detection |
| 3 | Detector and two-stage pipeline | Can a detector find the vessels and feed the classifier? | 87.4% found; found and correct 80.1 / 60.2 / 62.4% | Try free Sentinel-2 imagery |
| 4 | High-res models on Sentinel-2 (A) | Do they work at 10 m? | No: 0.3% of vessels found | Resample training data to 10 m |
| 5 | Resolution audit (B, measurements) | What resolution is each image really? | Recorded values incomplete and partly wrong; measured per source | Resample by measured resolution |
| 6 | Retrain at 10 m (B, C) | How well do models trained on resampled data do? | Detector mAP50 63.2; classifiers suspiciously high | Test on real Sentinel-2 |
| 7 | 10 m models on Sentinel-2 (D) | Does resampled training transfer? | No: 12,500 to 15,800 detections per scene | Pivot to real Sentinel-2 labels |
| 8 | Fine-tune on real Sentinel-2 (E) | Does a fine-tuned YOLO detect vessels at 10 m? | Yes: 93.1% found, comparable to the published model | Deploy it (F) |
| 9 | Baselines (step 10) | One table of every detector on Sentinel-2, stock YOLO included | not run yet | Write-up |
| 10 | Monitor (F) | Detector over a live area, with email alerts | not built | |

## 0. The 2025 study (course project, submitted 2025-12-11)

**Question.** How far do small CNNs trained from scratch get against a pretrained deep network on fine-grained ship classification, and how does that change from 4 to 48 classes? Motivation: tracking vessels that switch off or spoof AIS, such as the Russian shadow fleet, from satellite imagery.

**Data.** ShipRSImageNet: about 3,400 images and 17,500 vessels, label levels 0 (vessel / no vessel) to 3. Train 2,198 images, val 550; the test split (687 images) has no public labels.

**Setup** (from [`legacy/`](../legacy/) and the paper; the code is authoritative where they differ):

| | Net (small CNN) | Net_Max (VGG-style) | ResNet101 |
|---|---|---|---|
| Architecture | 3 conv layers (8, 16, 32 filters), 2 pooling, 1 fully connected | 4 blocks: 2 x (2 conv, 64 / 128 filters) and 2 x (3 conv, 256 / 512), batch norm, pooling | ImageNet weights, all layers fine-tuned |
| Optimiser | SGD, lr 0.001, momentum 0.9 | Adam, lr 0.001 (the paper says SGD; the code uses Adam, `legacy/level1.py:483`) | Adam, lr 0.0001 |
| Common | 30 epochs, batch 32, cross-entropy, 224 x 224 input (resize 256, random crop 224), flips, `RandomRotation(0.5)`; trained on Colab A100 / T4, about 25 to 30 minutes per model | | |

- Labels: one label per image, the majority class of its vessels.
- Class index built separately for train and val.
- Model selected and reported on the same val split; accuracy only; no seeds.
- Pipeline prototype: stock YOLO11n-OBB detects, crops are cut upright with PIL (the oriented box is lost), the level 2 ResNet101 classifies.

**Result** (best val accuracy):

| Level | Classes | Net | Net_Max | ResNet101 |
|---|---|---|---|---|
| 1 | 4 | 70.2 | 68.7 | 85.6 |
| 2 | 25 | 19.3 | 17.8 | 50.6 |
| 3 | 48 | 9.6 | 10.4 | 23.1 |

The paper calls the pipeline's results disappointing and blames the general-purpose pretrained models, the upright crops, and the one-label-per-image design. It names an R-CNN and an "out of the box vs. fine-tuned" comparison as next steps, and deploying on Sentinel-2 over the Baltic as the long-term goal.

**Evidence.** [`docs/Full Paper_Boppert.pdf`](Full%20Paper_Boppert.pdf), weights and histories in `5.Projects/models` (`best_model_{Net|Net_Max|ResNet}_Level{1,2,3}.pth`, `*_history_Level*.json`).

**Decision.** Rework the study before building on it (started 2026-10-03).

## 1. Audit and port (2026-10-03 to 2026-10-08; notebook steps 1 to 3)

**Question.** Are the 2025 weights complete, does a clean rewrite reproduce their scores, and what in the method was wrong?

**What was done.**
- Inventory of the Drive folder (step 1, 2026-10-03): all nine runs complete; each history's best val accuracy matches the paper. Extra files from earlier level 1 experiments identified (`final_model_Net_Max_deeper.pth` actually holds a small Net).
- Port into one config-driven package; the original study is kept behind two switches (`labels: image_majority`, `per_split_classes: true`) in [`configs/legacy/`](../configs/legacy/).
- Verification (step 2, 2026-10-08): all nine weights score within 0.01 points of the paper through the new code.
- Dataset check (step 3, 2026-10-08): test split unlabelled; every annotation has the four corners of its oriented box.

**Defects found.**

| Defect | Effect | Fix |
|---|---|---|
| One majority label per multi-ship image (`legacy/level1.py:60`) | Images with several vessels classified by one label | One sample per vessel, cut along its oriented box |
| Class index built per split (`legacy/level1.py:121`) | At level 3 train and val hold different classes, so correct predictions were scored as wrong | Index from train only, saved with each checkpoint |
| `ResNet.eval()` commented out in level 1 inference (`legacy/level1.py:826`) | Batch norm and dropout in training mode at test time | Eval mode in one shared evaluation |
| Selection and reporting on the same val split; accuracy only | Optimistic scores, no class-level view | Held-out test split, macro-F1, per class, by size, majority baseline |
| `RandomRotation(0.5)` is ±0.5°; a different optimiser per model; no seeds | Augmentation does nothing; models not comparable | Adam for all, seeds fixed |

**Result of the class-index fix alone.** Rescoring the same 2025 weights with the train index (`verify_legacy --shared-classes`, 2026-10-08): level 3 rises from 9.6 / 10.4 / 23.1 to 12.7 / 10.7 / 50.7. Levels 1 and 2 do not change. So the 2025 level 3 result was mostly an evaluation error.

**Evidence.** `5.Projects/models/inventory.json`, `legacy_verification.json`, `legacy_verification_shared_classes.json`.

**Decision.** Keep the three architectures and learning rates, change the data and evaluation, and rerun everything under one recipe.

## 2. Corrected classifier study (2026-10-08; notebook step 4)

**Question.** How do the three models compare when each vessel is cut out and classified on its own, with a held-out test set?

**Setup** ([`configs/crops/`](../configs/crops/)):
- One crop per annotated vessel, cut along its oriented box with 10% margin, resized to 224 x 224; flips as augmentation.
- Splits: the official val split (550 images, 3,103 vessels) became the test set; validation is 15% of train images (`val_fraction 0.15`, `split_seed 0`).
- Adam for all; lr 0.001 (Net, Net_Max), 0.0001 (ResNet101, ImageNet weights, all layers trained); 30 epochs; batch 32; seed 0; model chosen on val accuracy; test touched once.
- Level 3 has 50 classes here (the train file's categories), against 48 in 2025.

**Result** (test accuracy, macro-F1 in brackets):

| Level | Net | Net_Max | ResNet101 | Majority |
|---|---|---|---|---|
| 1 | 85.1 (74.2) | 84.2 (68.2) | 90.2 (82.2) | 38.0 |
| 2 | 59.5 (56.9) | 42.4 (26.0) | 68.4 (67.0) | 12.8 |
| 3 | 58.9 (70.1) | 38.5 (16.1) | 70.0 (78.4) | 12.8 |

- ResNet101 leads at every level. At level 3 it separates named warship classes almost perfectly and struggles with catch-all and small-boat classes (Other Ship, Other Merchant, Motorboat, Sailboat: F1 29 to 60).
- Level 3 by vessel size (ResNet101): 54% small (box under 32² px), 73% medium, 81% large.
- Leaving out `Dock` changes accuracy by under 2 points.

**Caveats on record.**
- Net_Max trained unstably (val accuracy swinging, still improving at epoch 30). Its row reflects the recipe, not the architecture. A rerun is undecided.
- Val ran 6 to 7 points above test at levels 2 and 3. Suspected cause: images are tiles of larger scenes, so neighbouring tiles land in both train and val. Not verified.
- Not directly comparable with 2025, which classified whole images.

**Evidence.** `5.Projects/runs/crops_level{1,2,3}_{net,net_max,resnet101}/` (weights, `history.json`, `metrics_val.json`, `metrics_test.json`, confusion matrices), `study_summary.json`.

**Decision.** The classifier is usable; the open problem from 2025 was several vessels per image. Build a detector that localises and let the classifier name each crop.

## 3. Detector and two-stage pipeline (2026-10-09; notebook step 5)

**Question.** Can a detector find the vessels, and how good is detect, cut out, classify end to end, compared with the 2025 prototype?

**Setup.**
- Detector: YOLO11s-OBB, pretrained on DOTA, fine-tuned on the level 0 boxes (vessel and dock, so it only localises); 100 epochs, image size 1024, batch 16, seed 0, same train / val / test split as the classifiers ([`configs/detect/yolo11s_obb.yaml`](../configs/detect/yolo11s_obb.yaml)).
- Pipeline: detections at confidence 0.25, each cut along its oriented box, classified by the step 2 ResNet101; a vessel counts as found at IoU of at least 0.5.

**Result.**
- Detector on test: precision 88.8, recall 85.3, mAP50 87.8, mAP50-95 68.2 (val about 5 points higher).
- Pipeline: 87.4% of vessels found, 78.3% of detections match a vessel.

| Level | Correct class, if found | Found and correct | Classifier on true boxes |
|---|---|---|---|
| 1 | 91.6 | 80.1 | 90.2 |
| 2 | 68.9 | 60.2 | 68.4 |
| 3 | 71.4 | 62.4 | 70.0 |

Detected boxes cost the classifier nothing; what the pipeline loses, it loses to missed vessels.

Comparison with the prototype, level 2:

| Detector | Crops and classifier | Found | Correct if found | Found and correct |
|---|---|---|---|---|
| Stock YOLO11n-OBB, conf 0.5 (2025 setup) | upright crops, ResNet101 trained on whole images | 56.1 | 19.9 | 11.2 |
| Stock YOLO11n-OBB | rotated crops, ResNet101 on crops | 65.5 | 60.3 | 39.5 |
| Stock YOLO11s-OBB | rotated crops, ResNet101 on crops | **Gap:** notebook 05c not run | | |
| Fine-tuned YOLO11s-OBB | rotated crops, ResNet101 on crops | 87.4 | 68.9 | 60.2 |

The stock models have no dock class (about 5% of test objects), which costs them a few points of recall. Without the 05c row, the gain from 65.5 to 87.4% found mixes fine-tuning with model size.

**Evidence.** `5.Projects/runs/yolo11s_obb/`, `pipeline_*.json`. The Colab output stopped displaying at epoch 96, but the run completed.

**Decision.** At high resolution the pipeline works. Next question: can it run on free imagery? Sentinel-2 (optical) chosen over Sentinel-1 (SAR, left as an extension), decided 2026-10-09. Plan A to F agreed the same day. Rule at the time: the monitor deploys Daniel's own models; published models are benchmarks.

## 4. High-resolution models on Sentinel-2 (step A, 2026-10-10; notebook step 6)

**Question.** Do models trained on 0.3 to 1 m imagery work on 10 m Sentinel-2?

**Benchmark.** Finnish Environment Institute vessel annotations (zenodo.org/records/15019034, CC BY 4.0): axis-aligned boxes around every vessel and its wake. Test set: seven scenes of tiles 34VEN and 34VER, 1,694 vessels. Imagery: L1C true colour (TCI) from the Copernicus Data Space. Scenes cut into 320 x 320 px chips, upsampled to 640 by the model, as the published model was trained. Reference: the published YOLO11s (`yolo11s_tci.pt`, AGPL-3.0, benchmark only). Daniel chose not to label any data.

**Scoring.** "Found": a detection's centre lies inside the vessel's box (IoU is brittle for boxes of a few pixels); also reported at IoU 0.5. "Hits": share of detections that found a vessel. Confidence 0.25.

**Result.**

| Detector | Detections | Found | Found (IoU 0.5) | Hits |
|---|---|---|---|---|
| High-res YOLO11s-OBB | 797 | 0.3% (5 vessels, all boxes over 100 m) | | 0.6% |
| Finnish reference | 2,822 | 90.1% | 86.5% | 54.1% |

Classifiers on the labelled boxes: level 1 said Merchant for all 1,694; level 3 said Motorboat 62%, Sailboat 38%. That is a near-constant answer and cannot be scored. The size-contradiction check flagged 0 of 16.

**What the imagery showed.** The labelled vessels are mostly small boats of a few bright pixels, often with a wake; boxes include the wake, so even 150 to 280 m boxes are small fast boats. The benchmark measures small-boat and wake detection, which differs from ShipRSImageNet (mostly moored ships).

**Evidence.** `5.Projects/runs/sentinel2_step_a.json`, `sentinel2_step_a_examples/`.

**Decision.** The high-resolution models do not transfer. Try to adapt them by training on ShipRSImageNet resampled to 10 m (step B).

## 5. Resolution audit (step B measurements, 2026-10-10; notebooks 7a, 7b)

**Question.** To resample to 10 m, each image's true resolution is needed. What is it?

**Result 7a.** `Img_Resolution` is recorded for only 1,257 of 2,748 labelled images (0.30 m: 421; 1.07 m: 819; 0.92 m: 9; 4.00 m: 8), leaving 6,829 of 13,963 vessels without one. The recorded values give impossible lengths (median warship 355 m, longest vessel 1,060 m).

**Result 7b.** Grouped by origin, resolution measured from vessels of named classes with known hull lengths:

| Source | Images | Vessels | Recorded | Measured (median, 10th to 90th percentile) |
|---|---|---|---|---|
| FGSD, Google Earth | 1,470 | 6,808 | none | 0.54 m (0.25 to 0.60) |
| HRSC, Google Earth | 819 | 2,720 | 1.07 m | 0.49 m (0.42 to 0.56); recorded value wrong by about 2.2x |
| xView, WorldView-3 | 421 | 4,397 | 0.3 m | no named classes; box sizes consistent with 0.3 m; trusted |
| JL-1, GF-2, Airbus | 38 | 38 | 0.92 / 4 m / none | too few |

**Decision** (agreed 2026-10-10). Resample each image with its own measured resolution where it holds a named vessel, the source median otherwise, and the recorded value only for xView, JL-1 and GF-2.

## 6. Retrain at 10 m (steps B and C, 2026-10-10; notebook 7c)

**Question.** How well can vessels be detected and classified at 10 m when the models are trained on ShipRSImageNet resampled to that resolution?

**Setup.**
- Area-averaged resampling per image to 10 m; vessels under 20 m left unlabelled.
- Classifiers: fixed-scale 56 px windows around each vessel, so size stays visible; otherwise the step 2 recipe ([`configs/crops_10m/`](../configs/crops_10m/)). 7,162 train and 1,259 val crops.
- Detector: at 10 m a whole scene is only 28 to 60 px, so training images are 320 px canvases composed from many resampled scenes at true scale (20 passes over the train scenes); YOLO11s-OBB from DOTA weights, 100 epochs, image size 640, batch 32 ([`configs/detect/yolo11s_obb_10m.yaml`](../configs/detect/yolo11s_obb_10m.yaml)).
- Step C scores classifier and detector separately on the resampled test split; they are not chained.

**Result.** Classifiers (test accuracy, macro-F1):

| Level | Net | Net_Max | ResNet101 | Majority |
|---|---|---|---|---|
| 1 | 79.7 (68.5) | 69.5 (36.3) | 87.7 (80.4) | 55.6 |
| 2 | 52.4 (41.1) | 18.8 (6.8) | 69.2 (56.6) | 9.4 |
| 3 | 52.9 (56.9) | 18.0 (4.7) | 66.2 (64.0) | 9.4 |

ResNet101 by hull length (2,100 test vessels of at least 20 m; under 50 m: 428, 50 to 150 m: 781, over 150 m: 891): level 1 75.0 / 88.0 / 93.5, level 2 47.4 / 68.4 / 80.5, level 3 44.9 / 65.9 / 76.8.

Detector on test canvases: precision 76.2, recall 59.2, mAP50 63.2, mAP50-95 41.9 (full resolution: 88.8 / 85.3 / 87.8 / 68.2).

**Why the classifier numbers are suspect.** The 56 px window is as large as a whole resampled scene, so each crop shows the scene with a black border, which reveals the source dataset; the sources differ in content (xView mostly small civilian boats, FGSD and HRSC mostly warships). **Gap:** notebook 7e tests this against a rule that knows only the source; not run. Until it runs, the 10 m classifier scores cannot be reported as vessel recognition.

**Evidence.** `5.Projects/runs/crops10m_*/`, `study_summary_10m.json`, `yolo11s_obb_10m/`.

**Decision.** Score the 10 m detector on real Sentinel-2 (step D). Expectation stated before the run: a modest rise from 0.3%, because the resampled data is mostly large moored ships and the Finnish vessels are small moving boats.

## 7. The 10 m models on Sentinel-2 (step D; notebook step 8)

**Question.** Does training on resampled data transfer to real Sentinel-2?

**Setup.** As step A, with the 10 m detector (dock class ignored), the high-resolution detector and the Finnish reference; the 10 m level 1 and level 3 ResNet101s tallied.

**Result.** First four scenes: 12,500 to 15,800 detections per scene against 66 to 450 labelled vessels. In the step E run on all seven scenes, the same detector made 85,762 detections, found 8.7% of vessels (0.4% at IoU 0.5), and 0.2% of its detections hit a vessel. Recall by box length: 0 of 750 under 50 m, 47 of 663 from 50 to 100 m, 101 of 281 over 100 m. **Gap:** the full step D table (`sentinel2_step_d.json`) is not on record; step 10b collects it.

**Decision** (Daniel, 2026-10-11). Resampling ShipRSImageNet to 10 m does not transfer to real Sentinel-2. Show the attempt and its failure in the write-up, then switch to real Sentinel-2 training data: the public Finnish labels. This replaces the rule that the monitor deploys only models trained on Daniel's own data; the deployed detector is his own training run on public labels. Also decided on 2026-10-11:
- Measurement (length, heading, size class) is dropped and explained in the write-up instead: a box edge is good to about one pixel, so lengths carry roughly ±10 to 20 m, most of a small boat; and the Finnish boxes include the wake.
- Step E is one run from `yolo11s.pt` (upright boxes); no second run from the 10 m detector.
- Step F deploys the detector alone: no classifier and no measurement.

## 8. Fine-tune on real Sentinel-2 (step E, 2026-10-16; notebook step 9)

**Question.** How well does a standard pretrained YOLO detect vessels in Sentinel-2 once it is fine-tuned on real labelled Sentinel-2 imagery?

**Setup** ([`configs/detect/yolo11s_finland.yaml`](../configs/detect/yolo11s_finland.yaml), [`detect/finland_export.py`](../src/vesselwatch/detect/finland_export.py)):
- Start: `yolo11s.pt` (COCO, upright boxes), single class.
- Scenes: train eight dates of tiles 34VEM, 34WFT, 35VLG; validation one further date of each (same coastlines, so it says nothing about new places); test the seven scenes of steps A and D (other tiles).
- Chips 320 x 320 px; two empty chips kept per chip with a vessel (seed 0) in train and validation; all chips in test.
- 100 epochs, image size 640, batch 32, seed 0; optimiser chosen by Ultralytics: AdamW, lr 0.002; mosaic off for the last 10 epochs; A100, 33 minutes. A first attempt was interrupted; the run was restarted from scratch.

| Split | Chips with vessels | Without | Vessels |
|---|---|---|---|
| Train | 1,526 | 2,863 | 4,732 |
| Validation | 722 | 1,444 | 2,440 |
| Test | 731 | 7,844 | 1,694 |

**Training.** Validation mAP50 rose from 0.07 (epoch 1) to about 0.80 by epoch 70 and stayed there (0.80 at epoch 100); mAP50-95 levelled off at about 0.33. No sign of overfitting. **Gap:** the test-chip line (`yolo11s_finland/metrics_test.json`) was cut off in the pasted output; step 10a prints it.

**Result on the seven test scenes** (confidence 0.25):

| Detector | Detections | Found | Found (IoU 0.5) | Hits |
|---|---|---|---|---|
| Fine-tuned YOLO11s | 4,065 | 93.1% | 89.8% | 38.8% |
| Finnish reference | 2,822 | 90.1% | 86.5% | 54.1% |
| 10 m YOLO11s-OBB | 85,762 | 8.7% | 0.4% | 0.2% |

| Detector | At 0.5: detections / found / hits | At 0.7 |
|---|---|---|
| Fine-tuned | 2,281 / 73.7% / 54.7% | 156 / 5.6% / 60.9% |
| Finnish reference | 1,612 / 61.2% / 64.3% | 23 / 0.6% / 43.5% |

| Detector | Under 50 m | 50 to 100 m | Over 100 m |
|---|---|---|---|
| Fine-tuned | 665 / 750 | 632 / 663 | 280 / 281 |
| Finnish reference | 619 / 750 | 627 / 663 | 280 / 281 |

**Reading.**
- At the same threshold the fine-tuned model finds more vessels, mostly small ones, and raises more false alarms.
- At equal precision the reference is ahead: about 54% hits costs the fine-tuned model a threshold of 0.5 and leaves 73.7% found; the reference has the same hit rate at 0.25 and finds 90.1%. Fair claim: comparable to the published model, more sensitive at the default threshold.
- Confidences are low on these targets; above 0.5 both models collapse. The monitor's threshold has to lie between 0.25 and 0.5.
- "Hits" counts unlabelled boats as false alarms; that affects both models alike.
- The 10 m row here kept the 10 m detector's dock class (the GitHub version of the notebook ran without `--drop-class Dock`); counts per scene are in the same range as step D.

**Evidence.** `5.Projects/runs/yolo11s_finland/`, `sentinel2_step_e.json`, `sentinel2_step_e_examples/`.

**Decision.** This detector goes into the monitor (step F). Before that, one baseline table (step 10).

## 9. Baselines (notebook step 10; not run yet)

**Why.** Steps A, D and E each scored some detectors with slightly different settings, and two comparisons are missing: stock vs. fine-tuned YOLO on Sentinel-2, and the fine-tuned model vs. the reference at equal precision.

**Setup.** All five detectors on the seven test scenes, confidence 0.25, dock class dropped, the stock model counting only `boat`: stock YOLO11s (COCO), high-res YOLO11s-OBB, 10 m YOLO11s-OBB, fine-tuned YOLO11s, Finnish reference. Also scored at 0.3 to 0.7, with the most vessels found while at least 50, 60 or 70% of detections are hits.

**Result.** **Gap:** not run.

## 10. Monitor (step F; not built)

Detector of step 8 over a live Sentinel-2 area, emailing detections, positions and a picture. No classifier, no measurement. Open: the area (Baltic waters suit the Finnish training data and free Danish AIS) and the alert threshold (0.25 finds more and alerts more; 0.5 misses about a quarter of vessels).

## Decisions log

| Date | Decision | Reason | By |
|---|---|---|---|
| 2026-10-03 | Rework the 2025 study in a tested package before extending it | Defects in labels, class index, eval mode and evaluation | Daniel |
| 2026-10-03 | Park the AIS-labelled type classifier ([note](experiments/ais-type-classifier.md)) | Not needed for the main line | Daniel |
| 2026-10-08 | Official val split becomes the test set; validation carved from train by image | Test split has no labels | forced by data |
| 2026-10-08 | One sample per vessel along its oriented box; class index from train; Adam for all | Fix the 2025 defects; make models comparable | Daniel |
| 2026-10-09 | Detector only localises (level 0); class comes from the crop classifier | Classifier already strong; detector trained on two classes | Daniel |
| 2026-10-09 | Sentinel-2 (optical) for the monitor; Sentinel-1 left as an extension | Same modality as the training data | Daniel |
| 2026-10-09 | Plan A to F; no labelling; published models only as benchmarks | Use only public labels | Daniel |
| 2026-10-10 | Resample by measured resolution, not the recorded one | Recorded values missing or wrong by 2.2x | Daniel, from 7a / 7b |
| 2026-10-10 | Fixed 56 px crops at 10 m; detector on composed 320 px canvases | Keep size visible; scenes too small at 10 m | Daniel |
| 2026-10-11 | Drop resampling; train on real Sentinel-2 (Finnish labels) | Step D: thousands of false alarms per scene | Daniel |
| 2026-10-11 | Drop measurement (length, heading, size class) | ±10 to 20 m per box; boxes include the wake | Daniel |
| 2026-10-11 | Step E: one run from `yolo11s.pt`; no second run from the 10 m detector | | Daniel |
| 2026-10-11 | Step F: detector only, email alerts with positions and a picture | | Daniel |
| not yet | Monitor area and alert threshold | | Daniel |

## Not done, on purpose or not yet decided

- Roadmap 11 (augmentation, class balancing, schedule, early stopping) and 14 (ResNet ablation: linear head, partial, full, from scratch): postponed.
- Net_Max rerun with a lower learning rate or more epochs: undecided.
- Validation split grouped by source scene, to test the 6 to 7 point val–test gap: not done.
- The fine-tuned high-resolution detector on outside high-resolution imagery: not done; only a stock detector was tried on a NAIP image of Norfolk.
- IMT Atlantique / CLS Sentinel-2 set (60 scenes, ships with length and heading): downloaded, not inspected.

## Open gaps before the write-up

| Gap | Closed by |
|---|---|
| Stock YOLO11s-OBB in the high-resolution pipeline (stock vs. fine-tuned at the same size) | notebook 05c |
| Do the 10 m classifiers recognise vessels or the source dataset? With the high-resolution classifiers as control | notebook 07e |
| Stock YOLO11s on Sentinel-2; all detectors in one table; comparison at equal precision | notebook 10a |
| Fine-tuned detector's scores on its test chips | notebook 10a (prints `metrics_test.json`) |
| Full step D table; training curves, configs and exact scores of every run | notebook 10b (`pathway_results.json`) |

Order for one Colab session: 05c, 07e, 10a, 10b. Everything is inference only. 05c and 07e copy or read ShipRSImageNet from Drive, which takes the longest.
