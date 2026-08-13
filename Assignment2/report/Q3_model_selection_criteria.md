# Q3 — Research Analysis & Model Selection

Citation numbers follow the reference list in [Q2](Q2_literature_review.md); the sources
used here are repeated at the end for convenience.

## 1. Comparative benchmark

Published figures first. The important thing to notice is the last column.

| Model | mAP@0.5:0.95 | Params (M) | FLOPs | Published latency | Measured on | Source |
|---|---|---|---|---|---|---|
| YOLOv8n | 37.3 | 3.2 | 8.7 G @640 | 80.4 ms ONNX / 0.99 ms TensorRT | CPU / **A100** | [2] |
| YOLO11n | 39.5 | 2.6 | 6.5 G @640 | 56.1 ms ONNX / 1.5 ms TensorRT | CPU / **T4** | [4] |
| YOLO26n | 40.9 | 2.4 | 5.4 G @640 | 38.9 ms ONNX / 1.7 ms TensorRT | CPU / **T4** | [12] |
| RT-DETR-R18 | 46.5 | 20 | 60.7 G @640 | 217 FPS (4.6 ms) | **T4** | [6] |
| YOLO-NAS-S | 47.5 | not published | not published | 3.21 ms | **not stated** | [7] |
| EfficientDet-Lite0 | 26.4 | 3.2 | not published | 36 ms | **mobile CPU** | [10] |

The published latencies were measured on an A100, three different T4 setups, an
unspecified device, and a phone. They cannot be ranked against each other. This is the
reason the brief asks for an independent benchmark, and it is why the scoring below uses
my own measurements rather than the numbers above.

Two figures are missing from the vendors entirely. Deci publish no parameter count, FLOPs
or model size for YOLO-NAS-S, and Google publish no FLOPs for the EfficientDet-Lite
variants. I measured both.

### My own measurements

Apple Silicon, batch size 1, mean over 100 iterations after 10 warm-up passes, identical
input image and confidence threshold of 0.4 for every model.

| Model | CPU latency (ms) | CPU FPS | MPS latency (ms) | Params (M) | Size (MB) | FLOPs |
|---|---|---|---|---|---|---|
| YOLOv8n | 111.7 ± 40.1 | 9.0 | 29.0 ± 6.6 | 3.15 | 12.61 | 8.7 G |
| YOLO11n | 87.3 ± 23.1 | 11.5 | 30.0 ± 6.0 | 2.62 | 10.46 | 6.5 G |
| YOLO26n | 88.4 ± 26.0 | 11.3 | 30.8 ± 8.2 | 2.41 | 9.64 | 5.4 G |
| RT-DETR-R18 | 434.4 ± 105.8 | 2.3 | not supported | 20.17 | 80.70 | 60.7 G |
| YOLO-NAS-S | 533.0 ± 251.9 | 1.9 | not tested | 19.05 | 76.22 | **33.9 G** (measured) |
| EfficientDet-Lite0 | 470.2 ± 90.6 | 2.1 | 80.4 ± 70.3 | 3.24 | 12.97 | **1.95 G @320** (measured) |

Notes on the table. Size is computed as parameters x 4 bytes rather than file size, because
an Ultralytics `.pt` bundles training metadata that a HuggingFace checkpoint does not, so
comparing file sizes would measure packaging rather than models. RT-DETR could not run on
MPS at all: `transformers` builds its position embedding in float64 and MPS has no float64.
YOLO-NAS-S runs in a separate Python 3.10 environment and was benchmarked on CPU only, so
CPU is the only column where all six models appear. EfficientDet-Lite0 runs at its native
320x320 while everything else runs at 640, so part of its low FLOPs count is input size
rather than architecture.

The standard deviations are large. YOLO-NAS-S at 533 ± 252 ms is a spread of nearly 50%,
and two runs of the identical RT-DETR configuration returned 503 ms and 434 ms. Differences
under roughly 15% in the CPU column should be treated as noise.

## 2. Scoring method

Four criteria, normalised to [0, 1] by min-max across the six models, with direction
handled per criterion so that higher is always better.

| Criterion | Weight | Why |
|---|---|---|
| Accuracy (published mAP) | 35% | A detector that misses objects is useless regardless of speed, so accuracy carries the largest single weight. It does not carry a majority because the brief's premise is that accuracy alone does not decide edge deployments. |
| Latency (my CPU measurement) | 30% | The target is real-time inference. CPU is used because it is the only device on which all six models run, and because one of the three hardware profiles in Q2 is CPU-only. |
| Model size | 20% | The Orin Nano shares 8 GB between the system and the model, and a phone shares less. |
| Deployment ease | 15% | A model that cannot be exported cannot ship. Scored from what each model actually cost me in this project, not from impressions. |

Deployment ease is the only subjective criterion, so its scoring is set out explicitly:

- **5 — YOLOv8n, YOLO11n, YOLO26n.** One-line export to ONNX, TensorRT and CoreML.
- **3 — EfficientDet-Lite0.** Runs, but tfhub.dev now returns 404, so the route named in
  the brief is dead. Needs manual pre-processing, manual box rescaling, and remapping from
  the COCO-90 id space.
- **2 — RT-DETR-R18.** Could not run on MPS at all. Heavier export path, and the only
  variant matching the brief lives in `transformers` rather than Ultralytics.
- **1 — YOLO-NAS-S.** Official weight host offline since the NVIDIA acquisition. Requires a
  separate pinned Python 3.10 environment, which in turn freezes numpy, onnx, onnxruntime
  and protobuf at 2023 releases.

## 3. Scoring table

| Model | mAP | Latency (ms) | Size (MB) | Deploy | s_acc | s_lat | s_size | s_deploy | **Total** |
|---|---|---|---|---|---|---|---|---|---|
| YOLO26n | 40.9 | 88.4 | 9.64 | 5 | 0.687 | 0.998 | 1.000 | 1.00 | **0.890** |
| YOLO11n | 39.5 | 87.3 | 10.47 | 5 | 0.621 | 1.000 | 0.988 | 1.00 | **0.865** |
| YOLOv8n | 37.3 | 111.7 | 12.61 | 5 | 0.517 | 0.945 | 0.958 | 1.00 | **0.806** |
| RT-DETR-R18 | 46.5 | 434.4 | 80.70 | 2 | 0.953 | 0.221 | 0.000 | 0.25 | **0.437** |
| YOLO-NAS-S | 47.5 | 533.0 | 76.22 | 1 | 1.000 | 0.000 | 0.063 | 0.00 | **0.363** |
| EfficientDet-Lite0 | 26.4 | 470.2 | 12.97 | 3 | 0.000 | 0.141 | 0.953 | 0.50 | **0.308** |

The two most accurate models finish fifth and fourth. That is the scoring method working as
intended rather than failing: YOLO-NAS-S wins accuracy outright at 47.5 mAP and still comes
last but one, because it is six times slower than YOLO11n, seven times larger, and cannot be
installed alongside the rest of the project.

A caveat on the method. Min-max normalisation is sensitive to outliers, and here RT-DETR and
YOLO-NAS at roughly 78 MB stretch the size axis so much that all three nano YOLOs compress
into the top 5% of it. Reweighting size downward would not change the ranking, since the
three leaders are separated mainly by accuracy and the gap to fourth place is 0.37, but the
method flatters the leaders more than the raw numbers justify.

## 4. Ranking and model carried forward

The ranking is YOLO26n, YOLO11n, YOLOv8n, RT-DETR-R18, YOLO-NAS-S, EfficientDet-Lite0.

**YOLO26n scores highest at 0.890**, driven by the lowest parameter count, the lowest FLOPs
and the fastest published CPU ONNX latency of the six. The brief permits substituting the
proposed additional model for one of the five, so carrying it forward would have been
allowed.

**I carried YOLO11n forward into Q4 to Q8 instead**, on engineering risk rather than score.
Q5 requires five separate export and optimisation paths: FP16, INT8 with calibration, ONNX
Runtime, TensorRT, and iterative pruning with retraining. YOLO11n has substantially more
mature support across all five, while YOLO26 was released in September 2025 and its export
paths are far less exercised. With a fixed deadline, a failure in one of those five paths
late in the work would have been unrecoverable. The cost of this decision is 1.4 mAP
(39.5 against 40.9) and 0.025 of composite score, which I judged a fair price for a
materially lower chance of an unresolvable export failure.

That judgement turned out to be load-bearing. The optimisation work in Q5 hit three
separate toolchain failures even on the well-supported model: `onnxruntime-gpu` replaced the
cuDNN that PyTorch links against and broke training entirely, TensorRT auto-installation
mid-session corrupted a running interpreter, and the ONNX Runtime CUDA provider never became
available at all, so that variant had to be measured on CPU and labelled as such.

## References

Numbering follows [Q2](Q2_literature_review.md).

- [2] G. Jocher, A. Chaurasia, and J. Qiu, Ultralytics YOLOv8 (Version 8.0.0) [Software]. 2023.
- [4] G. Jocher and J. Qiu, Ultralytics YOLO11 (Version 11.0.0) [Software]. 2024.
- [6] PekingU, "rtdetr_r18vd," Hugging Face model card.
- [7] Deci-AI, "YOLO-NAS," GitHub, 2023.
- [10] Google Research, AutoML: EfficientDet [Software]. GitHub, 2020.
- [12] Ultralytics, "Ultralytics YOLO26," Ultralytics YOLO Docs, 2026.
