# vessel-watch

Vessel detection and classification from satellite imagery, from a model-comparison study to a scheduled monitor on open Copernicus data.

**Status: work in progress.** The original 2025 study is preserved in [`legacy/`](legacy/). The package, the corrected experiments and the monitor are being built step by step; see the roadmap below.

## Motivation

Vessels that switch off or spoof AIS, such as the tankers of the Russian shadow fleet, cannot be tracked from their own transmissions. Satellite imagery is an independent source. This project asks two questions:

1. How far do small CNNs trained from scratch get against a pretrained deep network on fine-grained ship classification, and how does that change as the label set grows from 4 to about 50 classes?
2. What of this can be run automatically on free Sentinel imagery over an area of interest?

## Two tracks

| | Track A: classification study | Track B: Copernicus monitor |
|---|---|---|
| Data | [ShipRSImageNet](https://github.com/zzndream/ShipRSImageNet), 0.12–6 m optical, about 3,400 images and 17,500 annotated vessels, of which 2,748 images and 13,963 vessels have public labels | Sentinel-2, 10 m optical, via the Copernicus Data Space Ecosystem |
| Task | Classify vessels at three label levels (4 / 25 / ~50 classes); detect with YOLO-OBB | Detect vessels in new scenes over an AOI, estimate size and heading, email an alert |
| Models | Small CNN, VGG-style CNN, ResNet (frozen, partial and full fine-tune), YOLO11-OBB | One detector trained on Sentinel-resolution data, chosen on the Track A evidence (most likely YOLO) |

The tracks share one package and one predictor interface. They do not share weights: a ship that fills an image at 0.5 m is a few pixels at 10 m, so Sentinel-2 supports detection and coarse size class, not class-level identification.

## Results of the original study

Best validation accuracy after 30 epochs, one label per image:

| Level | Classes | Small CNN | VGG-style CNN | ResNet101 (fine-tuned) |
|---|---|---|---|---|
| 1 | 4 | 70.2% | 68.7% | 85.6% |
| 2 | 25 | 19.3% | 17.8% | 50.6% |
| 3 | 48 | 9.6% | 10.4% | 23.1% |

These numbers come with known problems, which the rework addresses: each multi-ship image was given a single majority label, the model was selected and scored on the same split, and only accuracy was reported.

## Roadmap

**Phase 0: inputs**
- [x] 1. Inventory of the trained weights and histories on Google Drive
- [ ] 2. Small local data sample for tests

**Phase 1: foundation**
- [x] 3. Repository and legacy code
- [x] 4. Project files
- [x] 5. Local environment
- [x] 6. Port the legacy scripts into one config-driven package
- [x] 7. Check the port reproduces the original numbers

**Phase 2: fix the method**
- [x] 8. Stable class mapping, correct eval mode, seeds
- [x] 9. Held-out test split
- [x] 10. Per-vessel crops instead of one label per image
- [ ] 11. Augmentation, class balancing, schedule, early stopping
- [x] 12. Macro-F1, per-class metrics, confusion matrices
- [ ] 13. Rerun all models at all levels under identical conditions

**Phase 3: pretrained models and detection**
- [ ] 14. ResNet ablation: linear head, partial, full fine-tune, from scratch
- [ ] 15. YOLO11-OBB trained end to end on the dataset
- [ ] 16. Two-stage detect, rotated crop, classify pipeline
- [ ] 17. Common predictor interface

**Phase 4: Copernicus monitor**
- [ ] 18. Copernicus Data Space client for an AOI
- [ ] 19. Sentinel-resolution detector, using the model family that Track A shows makes most sense
- [ ] 20. Scene check, tiling, inference, georeferencing, deduplication
- [ ] 21. Email alerts
- [ ] 22. Scheduled runs
- [ ] 23. Validation on archived scenes

**Phase 5: extensions**
- [ ] 24. Sentinel-1 SAR and AIS cross-check
- [ ] 25. Map dashboard
- [ ] 26. Final write-up

## Parked experiments

Ideas written down for later; not part of the roadmap yet.

- [Vessel type from Sentinel-2 with labels from archived AIS](docs/experiments/ais-type-classifier.md)

## Setup

```bash
pip install -e ".[dev]"
pytest
```

The dataset and trained weights are not in this repository. Point `VESSELWATCH_DATA` at a `ShipRSImageNet_V1` folder, then:

```bash
python -m vesselwatch.train --config configs/crops/level1_net.yaml
python -m vesselwatch.evaluate --config configs/crops/level1_net.yaml --weights runs/crops_level1_net/best.pt --split test
python -m vesselwatch.study --configs configs/crops    # all nine runs, each scored once on test
```

`configs/crops/` holds the corrected experiments: one sample per annotated vessel, cut out along its oriented box. The dataset's test labels are not public, so its val split serves as the held-out test set and validation is carved out of train by image. `configs/legacy/` reproduces the original study.

Full training runs on Google Colab with the data on Google Drive, through [`notebooks/colab_study.ipynb`](notebooks/colab_study.ipynb).

## Reference

Zhang, Z., Zhang, L., Wang, Y., Feng, P., & He, R. (2021). ShipRSImageNet: A large-scale fine-grained dataset for ship detection in high-resolution optical remote sensing images. *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, 14, 8458–8472.

## Author

Daniel Boppert
