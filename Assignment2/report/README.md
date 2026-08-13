# Technical Report - Questions 1 to 12

Index to the answers. Each question links to where the work and the numbers live.

Repository: https://github.com/SanjuktaK/Atri_ai_assignment
wandb project: [wandb](https://wandb.ai/trishab/edge-vision-yolo11n/workspace?nw=nwusertrishab)

Selected model carried into Q4 to Q8: **YOLO11n**.

Measurement environments differ by section and are not interchangeable. Q1, Q3 and Q4 ran
locally on Apple Silicon (CPU/MPS). Q5 to Q12 ran on a Colab A100. 

---

### Q1 - Environment & Baseline Sanity Check
[`model_preview.ipynb`](../model_preview.ipynb)

Six models run on one COCO val2017 street scene (`000000212573`, containing person, car,
traffic light and umbrella) at a shared confidence threshold of 0.4. Annotated outputs are
in [`output/`](../output) as `q1_<model>.jpg`, with all six side by side in
`q1_all_models.jpg`. A summary table at the top of the notebook records library, input
size, NMS-free status, parameters, size, latency and detection count per model.

The five required models plus YOLO26n as the additional proposal. Detection counts on the
same image: RT-DETR 12, YOLO-NAS-S 11, YOLOv8n 10, YOLO11n 8, YOLO26n 8,
EfficientDet-Lite0 6.

### Q2 - Literature Review
[`Q2_literature_review.md`](Q2_literature_review.md)

Design-intent paragraph per model, the comparative table (architecture family, parameters,
FLOPs, anchor type, NMS-free, backbone, head), strengths and weaknesses with IEEE
citations, and deployment suitability across Jetson Orin Nano, Raspberry Pi 5 and Android
NNAPI/TFLite.

Two FLOPs figures are measured rather than cited, because no vendor publishes them:
YOLO-NAS-S at 33.9 G and EfficientDet-Lite0 at 1.95 G. The method reproduces published
figures to within 1%.

### Q3 - Research Analysis & Model Selection
[`Q3_model_selection_criteria.md`](Q3_model_selection_criteria.md)

Published benchmark table with per-row citations, own CPU/MPS measurements, the weighted
scoring method (accuracy 35%, latency 30%, size 20%, deployment ease 15%) with each weight
justified, and the full scoring table.

YOLO26n scored highest at 0.890. YOLO11n was carried forward instead, on engineering-risk
grounds set out in that document.

### Q4 - Proof of Concept
[`run_demo.py`](../run_demo.py) · [`model_preview.ipynb`](../model_preview.ipynb) ·
clip: [`output/demo_clip.mp4`](../output/demo_clip.mp4) and
[`demo_clip.gif`](../output/demo_clip.gif)

```bash
python run_demo.py --source data/demo.mp4 --model yolo11n.pt --conf 0.4 \
                   --output output/annotated_out.mp4
```

Boxes with class and confidence, a rolling 30-frame FPS counter top-left, mp4 written at
the input resolution and frame rate, and an end-of-run summary. On the 647-frame test
video: 19.93 average FPS, 47.85 ms average inference latency. The 15 second clip is cut
from 34.3 s, where all four classes appear.

### Q5 - Model Optimization & Pruning
[`Assignment2.ipynb`](../Assignment2.ipynb)

`benchmark()` is the shared export-and-measure pipeline every variant plugs into. Variants:
FP16, INT8 (250-image calibration set, above the 200 required), ONNX Runtime, TensorRT.
Lottery Ticket pruning runs three rounds of magnitude pruning with weight rewinding and
retraining at 20%, 50% and 70% sparsity.

COCO train2017 is 19 GB, so val2017 is split 4000/1000 and all pruning accuracy is
reported on the held-out 1000.

### Q6 - Benchmarking
[`Assignment2.ipynb`](../Assignment2.ipynb) · wandb report

mAP, FPS, latency mean and standard deviation, model size, GPU and CPU utilisation and
startup time, each over 100 iterations after 10 warm-up passes. The three required panels
(FPS bar, latency bar, accuracy vs model size scatter) are in the wandb report.

### Q7 - Analysis & Insights
wandb report

Seven observations and the production recommendation, based on the Q6 panels. Headline:
FP16 is the variant to ship, at 7.51 ms and 133.24 FPS against the baseline's 14.37 ms and
69.57 FPS, with mAP unchanged.

### Q8 - Loss Function Comparison
[`Assignment2.ipynb`](../Assignment2.ipynb)

Ultralytics uses `BCEWithLogitsLoss` rather than softmax cross-entropy, so this compares
BCE against focal-BCE. Per-epoch mAP curves and a final-mAP bar chart for both.

BCE 0.3755 against focal 0.3668. BCE leads at every epoch, not only at the end.

### Q9 - Hybrid Architecture
[`Assignment2.ipynb`](../Assignment2.ipynb) · architecture in `yolo11n-aifi.yaml`

YOLO11n with RT-DETR's AIFI encoder replacing YOLO11's C2PSA block at P5. Both the hybrid
and a control trained 10 epochs on identical data, so architecture is the only difference.

| | Params | Latency | FPS | mAP@0.5:0.95 |
|---|---|---|---|---|
| Control (YOLO11n) | 2.62 M | 19.11 ms | 52.3 | 0.3721 |
| AIFI hybrid | 3.16 M (+20.6%) | 20.19 ms (+5.7%) | 49.5 | 0.2669 |

Latency scales far more slowly than parameters. Accuracy does not hold: AIFI is the one
block that cannot inherit pretrained weights, and at 3 epochs the gap was 16 points
against 10.5 here, so the deficit is convergence rather than architecture.

### Q10 - Additional Pruning Technique
[`Assignment2.ipynb`](../Assignment2.ipynb)

Magnitude pruning with fine-tuning and no rewinding, run under the identical protocol to
the Q5 lottery-ticket rounds. Only one variable changes: each round keeps the previous
round's trained weights instead of rewinding to the originals, so any difference between
the two accuracy-vs-sparsity curves is attributable to rewinding alone. Both curves are
plotted together in the notebook.

### Q11 - Repository & Code Quality
[Top-level README](../../README.md)

Setup and reproduction steps, the two local environments and why they are separate, the
YOLO-NAS weight workaround, and the model-source deviations from the brief. Every number
in this report comes from a run logged to the wandb project.

### Q12 - Transfer Task
[`Assignment2.ipynb`](../Assignment2.ipynb)

Two of five models tested for low-light indoor deployment: YOLO11n (the Q3 winner) and
RT-DETR-R18 (best small-object recall in Q1). Evaluated on the same 1000-image held-out
split darkened to 0.25x gain.

| | mAP normal | mAP dark | drop |
|---|---|---|---|
| YOLO11n | 0.3982 | 0.3691 | 7.31% |
| RT-DETR-R18 | 0.4751 | 0.4378 | 7.84% |

Both degrade by almost the same fraction, so RT-DETR is not more robust to low light, it
is simply more accurate throughout. That refutes the hypothesis the model was selected on.
