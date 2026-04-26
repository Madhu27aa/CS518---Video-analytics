# Real-Time Open-Vocabulary Prompt-to-Mask Video Analytics
### Using YOLO-World and SAM-2 · CS 518 — Deep Learning for Computer Vision

**Team:** Madhumitha Seshaiah · Sumukh Govinda · Vishaal Dayashanker

---

## Project Overview

This project builds an open-vocabulary drone-view object detection and tracking system. A user specifies object categories in plain English — no retraining required for new categories. The system detects those objects in aerial drone imagery, produces pixel-accurate segmentation masks, and tracks them temporally across video frames.

**The core problem:** Standard object detectors trained on ground-level imagery fail on drone footage. Objects are tiny (67% of VisDrone bounding boxes are smaller than 32×32 pixels), densely packed, and viewed from angles that do not exist in general pretraining data.

**The solution:** Fine-tune YOLO-World (open-vocabulary detector) on VisDrone's aerial imagery so it learns domain-specific visual representations, then chain it with SAM-2 for segmentation and temporal tracking.

---

## Key Results

| Metric | Pretrained | Fine-tuned | Improvement |
|---|---|---|---|
| mAP@0.5 (full val) | 0.1036 | 0.2874 | **+177%** |
| mAP@0.5 (stress set) | 0.0882 | 0.2900 | **+229%** |
| mAP-small (<32px) | 0.0504 | 0.1606 | **+219%** |
| mAP@0.5:0.95 | 0.1000 | ~0.16 | **+60%** |
| SAM-2 mask IoU | — | 0.8085 | — |
| Tracking coverage | — | 100% | — |
| YOLO-World FPS | 34.3 | 36.0 | Real-time ✅ |

All 11 VisDrone classes improved after fine-tuning. Two classes that were completely undetectable by the pretrained model (motor, people) reached AP of 0.312 and 0.229 respectively after fine-tuning.

**Ablation finding:** Fine-tuning accounts for 69% of the total improvement. Prompt engineering alone (better text prompts on the pretrained model) accounts for only 31%. Domain adaptation through fine-tuning is essential — prompts alone cannot bridge the aerial domain gap.

---

## Pipeline Architecture

```
Text Prompt (e.g. "pedestrian, car, bus")
         │
         ▼
   CLIP Text Encoder
         │
         ▼
  YOLO-World Detector ──── Frame 0
  (fine-tuned on VisDrone)
         │
    Bounding Boxes
         │
         ▼
  SAM-2 Image/Video Predictor
         │
    Segmentation Masks
         │
         ▼
  Temporal Propagation (SAM-2 VideoPredictor)
         │
    Tracked Masks across all frames
         │
         ▼
    Annotated Output Video
```

---

## Datasets

| Dataset | Role | Size |
|---|---|---|
| **VisDrone DET** | Primary — fine-tuning + evaluation | 6,471 train / 548 val / 1,610 test images |
| **COCO 2017** | SAM-2 segmentation baseline evaluation | 330K images, filtered to 6 overlapping classes |
| **YouTube-VIS** | Video tracking — SAM-2 VideoPredictor evaluation | 2,985 videos, 90,160 frames |

### VisDrone Classes (11)
`pedestrian` · `people` · `bicycle` · `car` · `van` · `truck` · `tricycle` · `awning-tricycle` · `bus` · `motor` · `others`

---

## Project Structure

```
DLCV_OV_Analytics/
│
├── configs/
│   └── config.json                    # Single source of truth for all paths
│
├── checkpoints/
│   ├── yoloworld/
│   │   ├── yolov8l-world.pt           # Pretrained YOLO-World checkpoint
│   │   └── yolov8l-world-finetuned.pt # Fine-tuned on VisDrone (Milestone 3)
│   └── sam2/
│       └── sam2.1_hiera_large.pt      # SAM-2.1 Large pretrained checkpoint
│
├── datasets/
│   ├── raw/
│   │   ├── VisDrone/                  # Raw VisDrone splits
│   │   ├── COCO/                      # Raw COCO 2017
│   │   └── Youtube VIS/               # Raw YouTube-VIS
│   └── processed/
│       ├── visdrone/                  # YOLO-format labels + yaml + stress_set.json
│       ├── coco/                      # Filtered COCO labels (6 classes)
│       └── youtube_vis/               # Per-frame labels + video_index.json
│
├── outputs/
│   ├── metrics/                       # All JSON and CSV metric files
│   ├── tables/                        # Comparison tables
│   ├── visualizations/                # All charts and detection grids
│   └── videos/                        # Annotated demo MP4
│
└── notebooks/
    ├── 00_setup.ipynb
    ├── 01_dataset_audit.ipynb
    ├── 02_preprocess_visdrone.ipynb
    ├── 03_preprocess_coco.ipynb
    ├── 04_preprocess_ytvis.ipynb
    ├── 2_baseline_evaluation.ipynb
    ├── 03_finetune_yoloworld.ipynb
    ├── 04_video_tracking.ipynb
    ├── 5_final_pipeline.ipynb
    └── NLP_model.ipynb
```

---

## Notebooks — What Each Does

### Milestone 1 — Data Pipeline and Preprocessing

**`00_setup.ipynb`**
Mounts Google Drive, defines all project paths, clones YOLO-World and SAM-2 repos, installs dependencies, downloads model checkpoints, and saves `config.json`. Run this at the start of every new Colab session.

**`01_dataset_audit.ipynb`**
Verifies all three raw datasets are intact. Counts images vs annotation files, flags orphan pairs, confirms zero missing annotations. Ensures preprocessing does not silently propagate corrupted data.

**`02_preprocess_visdrone.ipynb`**
Converts VisDrone's CSV annotations to YOLO format. Handles ignored regions (category 0), degenerate boxes, and out-of-bounds coordinates. Builds `stress_set.json` — 5,906 images with more than 10 small objects each, used as the hard evaluation subset.

**`03_preprocess_coco.ipynb`**
Filters COCO 2017 to 6 classes overlapping with VisDrone, remaps non-sequential COCO category IDs to 0-based YOLO IDs, writes label files. Uses local SSD staging + tar transfer to avoid Drive FUSE flush failures.

**`04_preprocess_ytvis.ipynb`**
Extracts per-frame YOLO label files from YouTube-VIS's video-structured annotations. Handles `None` entries (object absent in frame), builds `video_index.json` for fast video lookup, saves `category_map.json`.

---

### Milestone 2 — Baseline Evaluation

**`2_baseline_evaluation.ipynb`**
Three sections:
- **Section A:** Evaluates pretrained YOLO-World on VisDrone val. mAP@0.5 = 0.1036 full val, 0.0882 stress set. This is the baseline all future results are compared against.
- **Section B:** Evaluates SAM-2 image predictor on 100 COCO val images using GT boxes as prompts. Mean mask IoU = 0.8085. Establishes segmentation ceiling.
- **Section C:** Runs the unified detect→segment pipeline on VisDrone stress-set images. Qualitative demonstration only (VisDrone has no segmentation GT).

---

### Milestone 3 — Fine-tune YOLO-World

**`03_finetune_yoloworld.ipynb`**
- Analyses VisDrone class imbalance (car 42%, pedestrian 23%, others <5%)
- Copies training data to local Colab SSD to avoid Drive I/O bottleneck during training
- Fine-tunes YOLO-World for 50 epochs on VisDrone with AdamW optimiser, lr=0.001, mosaic augmentation, Drive checkpoint backup every epoch
- Evaluates fine-tuned model using identical evaluator as Milestone 2
- Generates pretrained vs fine-tuned comparison table and charts
- Saves `yolov8l-world-finetuned.pt` to Drive

**Training configuration:**

| Parameter | Value | Reason |
|---|---|---|
| epochs | 50 | Model still improving at epoch 49 |
| batch | 8 | Safe for T4 15GB VRAM |
| optimizer | AdamW | Better than SGD for fine-tuning |
| lr0 | 0.001 | Lower than default to prevent catastrophic forgetting |
| mosaic | 1.0 | Increases rare-class exposure per batch |
| flipud | 0.5 | Valid for aerial view (no inherent up/down) |
| degrees | 0.0 | Drone footage is always roughly upright |
| close_mosaic | 10 | Disable mosaic last 10 epochs for convergence |

---

### Milestone 4 — Video Tracking Pipeline

**`04_video_tracking.ipynb`**
- Selects 10 YouTube-VIS training videos (10–36 frames, verified on disk)
- Runs fine-tuned YOLO-World on frame 0 of each video to generate box prompts
- Uses `SAM2VideoPredictor.init_state()` + `add_new_points_or_box()` + `propagate_in_video()` for temporal tracking
- 6/10 videos tracked successfully, all with 100% coverage
- 4 videos skipped: contained non-VisDrone categories (animals) outside the fine-tuned model's class set — a domain boundary finding, not a failure

---

### Milestone 5 — Final Pipeline and Report

**`5_final_pipeline.ipynb`**
- Aggregates all metrics from all milestones into a master summary
- Generates 4 publication-quality comparison charts
- Runs the complete end-to-end pipeline on demo video `6a80d2e2e5` (11 objects, 20 frames)
- Exports annotated MP4 video
- Generates final project report card (`final_report_card.json`)

---

### Milestone 6 — Extended Evaluation and NLP Demo

**`NLP_model.ipynb`**
Addresses three gaps from the original proposal:

**Gap 5 — Ablation Study**
Tests three conditions on 548 val images to isolate what drives the improvement:
- A: Pretrained + generic prompts → mAP 0.0278
- B: Pretrained + VisDrone-specific prompts → mAP 0.1071
- C: Fine-tuned + VisDrone classes → mAP 0.2874

Result: Fine-tuning accounts for 69% of total improvement. Prompt engineering alone is insufficient for aerial domain adaptation.

**Gap 3 — Extended Metrics**
- mAP-small: 0.0504 → 0.1606 (+219%) — directly proves small object challenge was addressed
- mAP@0.5:0.95: 0.1000 → ~0.16
- FPS: 34.3 → 36.0 (YOLO-World alone exceeds 30fps real-time threshold)

**Gap 1 — NLP Prompting Demo**
Shows the same pretrained model detecting different objects from different text prompts with no retraining. Demonstrates open-vocabulary capability and its boundaries (concrete nouns work; abstract phrases like "object in motion" do not — a finding, not a failure).

---

## Setup and Reproduction

### Requirements

```
Python 3.12
torch 2.10.0+cu128
ultralytics==8.3.145
hydra-core
iopath
submitit
einops
opencv-python-headless
pycocotools
supervision
albumentations
```

### Environment

All experiments run on Google Colab. The project is designed around Google Drive storage — all checkpoints, datasets, and outputs persist on Drive across sessions.

**Every new Colab session requires running `00_setup.ipynb` first** to remount Drive, clone repos to local `/content/`, and reinstall packages.

### Reproduction Steps

```
1. Run 00_setup.ipynb            — environment + checkpoints
2. Run 01_dataset_audit.ipynb   — verify raw data
3. Run 02_preprocess_visdrone.ipynb
4. Run 03_preprocess_coco.ipynb
5. Run 04_preprocess_ytvis.ipynb
   → Milestone 1 complete

6. Run 2_baseline_evaluation.ipynb
   → Milestone 2 complete

7. Run 03_finetune_yoloworld.ipynb   (2–3 hours training)
   → Milestone 3 complete

8. Run 04_video_tracking.ipynb
   → Milestone 4 complete

9. Run 5_final_pipeline.ipynb
   → Milestone 5 complete

10. Run NLP_model.ipynb
    → Milestone 6 complete
```

### Important: Drive Write Pattern

All notebooks use a local SSD staging pattern to avoid Google Drive FUSE flush failures:

```python
# All file writes go to local /content/staging/ first
# Then transferred to Drive via tar + cp + sync
subprocess.run("tar -czf outputs.tar.gz -C /content/staging .", shell=True)
subprocess.run("cp outputs.tar.gz /content/drive/.../ && sync", shell=True)
subprocess.run("tar -xzf .../outputs.tar.gz -C .../outputs/", shell=True)
```

Direct writes to Drive via `shutil.copy2()` or `open()` silently fail on large transfers because the FUSE buffer does not flush to Drive storage before the session ends.

---

## Models

### YOLO-World (yolov8l-world.pt)
- Architecture: YOLOv8-Large backbone + CLIP text encoder
- Pretraining: Objects365, COCO, CC3M
- Open-vocabulary: `set_classes(["any", "text", "prompt"])` at inference
- Fine-tuned checkpoint: `yolov8l-world-finetuned.pt` (91 MB)

### SAM-2.1 Large (sam2.1_hiera_large.pt)
- Architecture: Hiera image encoder + prompt encoder + mask decoder + memory module
- Pretraining: SA-1B (images) + SA-V (videos)
- Used pretrained only — no fine-tuning applied
- Image predictor: single-frame box-prompted segmentation
- Video predictor: temporal mask propagation across frames

---

## Key Engineering Decisions

**Why local SSD for training:** Reading 6,471 images from Drive repeatedly across 50 epochs would add 1–2 hours of I/O overhead. Data was copied once to `/content/visdrone/` and training read from there.

**Why not fine-tune SAM-2:** SAM-2 achieved 0.8085 mask IoU and 100% tracking coverage pretrained. Fine-tuning would require per-pixel segmentation ground truth (VisDrone has none) and a custom training loop. The pretrained model was sufficient for the project goals.

**Why single-phase training instead of proposed multi-phase:** Multi-phase training was designed for a multi-dataset setup. With VisDrone as the sole training dataset, a single run with low learning rate and AdamW achieved the same catastrophic-forgetting prevention that phased training was meant to provide.

**Why VisDrone for training instead of test-only:** The original proposal planned VisDrone as test-only with Objects365 as the training set. Objects365 (600 GB) was not feasible to download. VisDrone was promoted to training data, which produced stronger domain-specific results than the original plan would have.

**Why `set_classes()` requires short nouns not sentences:** YOLO-World uses CLIP text embeddings for region matching. CLIP maps short category nouns to tight embeddings in visual-semantic space. Sentences map to scene-level embeddings that do not match individual bounding box regions. "Person" works; "person walking" does not.

---

## Limitations

**Not real-time end-to-end:** YOLO-World alone runs at 36 FPS (real-time). SAM-2 video propagation runs at approximately 0.3 FPS on T4 GPU. The full pipeline is not suitable for real-time deployment without architectural optimization (e.g., lighter SAM-2 variant, keyframe-only detection).

**Domain-locked after fine-tuning:** The fine-tuned checkpoint is specialised for VisDrone's 11 classes. Unlike the pretrained model, it cannot be easily retargeted to arbitrary categories via `set_classes()`. The open-vocabulary flexibility is intentionally traded for domain accuracy.

**YouTube-VIS category mismatch:** 4 of 10 test videos contained non-VisDrone categories (animals) that the fine-tuned model cannot detect. This demonstrates the model's domain boundary and is a finding, not a failure.

**mAP@0.5:0.95 evaluated on subset:** The COCO standard metric was computed on 100 images (not all 548) due to the computational cost of 10 evaluation passes. Results are representative but not exact.

---

## Results Summary

### Detection — Per-Class AP@0.5

| Class | Pretrained | Fine-tuned | Δ |
|---|---|---|---|
| pedestrian | 0.0909 | 0.3339 | +0.2430 |
| people | 0.0000 | 0.2288 | +0.2288 |
| bicycle | 0.0909 | 0.1412 | +0.0503 |
| car | 0.3397 | 0.6129 | +0.2732 |
| van | 0.1408 | 0.3555 | +0.2147 |
| truck | 0.1343 | 0.2753 | +0.1410 |
| tricycle | 0.0909 | 0.2112 | +0.1203 |
| awning-tricycle | 0.0350 | 0.1360 | +0.1010 |
| bus | 0.2170 | 0.5094 | +0.2924 |
| motor | 0.0000 | 0.3120 | +0.3120 |
| others | 0.0000 | 0.0455 | +0.0455 |
| **mAP@0.5** | **0.1036** | **0.2874** | **+0.1838** |

### Ablation — What Drives the Improvement?

| Condition | mAP@0.5 | Share of gain |
|---|---|---|
| Pretrained + generic prompts | 0.0278 | baseline |
| Pretrained + VisDrone prompts | 0.1071 | 31% from prompts |
| Fine-tuned + VisDrone classes | 0.2874 | 69% from fine-tuning |

### Segmentation (SAM-2, box-prompted on COCO val)

| Class | Mean Mask IoU |
|---|---|
| bus | 0.8674 |
| car | 0.8507 |
| truck | 0.8432 |
| person | 0.8061 |
| motorcycle | 0.7177 |
| bicycle | 0.5988 |
| **Mean** | **0.8085** |

### Video Tracking (SAM-2 VideoPredictor on YouTube-VIS)

- Videos tracked: 6
- Mean tracking coverage: 100%
- Max simultaneous objects: 11
- Mean objects per video: 3.0

---

## References

1. Zhu et al., *Detection and Tracking Meet Drones Challenge*, IEEE TPAMI, 2021.
2. Lin et al., *Microsoft COCO: Common Objects in Context*, ECCV, 2014.
3. Yang et al., *Video Instance Segmentation*, ICCV, 2019.
4. Cheng et al., *YOLO-World: Real-Time Open-Vocabulary Object Detection*, CVPR, 2024.
5. Ravi et al., *SAM 2: Segment Anything in Images and Videos*, arXiv:2408.00714, 2024.

---

