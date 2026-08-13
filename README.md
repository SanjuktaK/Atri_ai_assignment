# Assignment 1

The report can be found under
Assignment1/Assignment1.ipynb
and the [wandb report](https://api.wandb.ai/links/trishab/8xad8n5m)



# Assignment 2 - Investigating Small Vision Models for Efficient Edge AI Deployment

Applied ML research assignment comparing five lightweight object detectors for real-time, resource-constrained edge deployment.

wandb project: [wandb](https://api.wandb.ai/links/trishab/pnk3mw1e)

### Where everything is

| What | Where |
|---|---|
| Q1 baseline inference, Q3 benchmark and scoring, Q4 demo | `Assignment2/model_preview.ipynb` |
| Q5 to Q12 (optimization, pruning, hybrid, transfer task) | `Assignment2/Assignment2.ipynb` (Colab) |
| Q2 literature review | `Assignment2/report/Q2_literature_review.md` |
| Q3 model selection | `Assignment2/report/Q3_model_selection_criteria.md` |
| Q1 annotated outputs, one per model | `Assignment2/output/q1_<model>.jpg`, side by side in `q1_all_models.jpg` |
| Q4 demo clip | `Assignment2/output/demo_clip.mp4` and `demo_clip.gif` |
| Q4 demo script | `Assignment2/run_demo.py` |
| YOLO-NAS subprocess wrapper | `Assignment2/yolo_nas_infer.py` |

### Environments

Two environments were used. The split is not a preference, it is forced by a hard version conflict, documented below.

| Environment | Python | Used for | Key packages |
|---|---|---|---|
| `.venv` (local) | 3.12.11 | Q1 inference, Q3 CPU/MPS benchmarks, Q4 demo | torch 2.13.0, torchvision 0.28.0, numpy 2.5.2, ultralytics 8.4.117, transformers 5.15.0, timm 1.0.28, effdet 0.4.1, onnx 1.22.0, onnxruntime 1.28.0, opencv-python 5.0.0.93, pycocotools 2.0.11 |
| `.venv-nas` (local) | 3.10.19 | YOLO-NAS-S only | super-gradients 3.7.1, torch 2.1.2, torchvision 0.16.2, numpy 1.23.0, onnx 1.15.0, onnxruntime 1.15.0, protobuf 3.20.3, torchmetrics 0.8.0, opencv-python 4.11.0.86 |

Exact versions for both are pinned in `requirements.lock.txt` and `requirements-yolonas.lock.txt`.

Q1, Q3 and Q4 were run locally (Apple Silicon, CPU/MPS). Q5 to Q10 and Q12 require CUDA and were run on a Colab A100. All latency and FPS figures in those sections are A100 measurements and are not comparable to the local numbers in Q3.

### Why two local environments

`super-gradients` 3.7.1 (the latest release, unchanged since 2024) pins `numpy<=1.23`,
while `ultralytics` requires `numpy>=1.23.0`. The two intersect at exactly one version,
`numpy==1.23.0`, which ships no wheels beyond CPython 3.10. Any environment containing
super-gradients is therefore pinned to Python 3.10 and, transitively, to
`onnx==1.15.0`, `onnxruntime==1.15.0`, `protobuf==3.20.3` and `torchmetrics==0.8`,
all mid-2023 releases. `onnxruntime` no longer publishes cp310 wheels at all.

Admitting super-gradients to the main environment would have frozen the entire project
at that stack, including the ONNX Runtime and INT8 work in Q5 and the pruning in Q10.
Isolating it costs one extra `pip install` and keeps everything else current.
YOLO-NAS-S is invoked from the main notebook as a subprocess (`yolo_nas_infer.py`),
so the analysis stays in one place.

The Ultralytics YOLO-NAS integration was evaluated as an alternative to a second
environment, but `ultralytics/models/nas/model.py` imports `super_gradients`
unconditionally, so it inherits the same `numpy<=1.23` constraint. It offers a different
API to the same dependency, not a way around it.

### Model sources, and why they differ from the brief

- **YOLOv8n, YOLO11n, YOLO26n**: `ultralytics`, as suggested.
- **RT-DETR (ResNet-18)**: `transformers` (`PekingU/rtdetr_r18vd`). The Ultralytics
  RT-DETR wrapper ships only `rtdetr-l` (ResNet-50) and `rtdetr-x` (ResNet-101); there is
  no R18 checkpoint. `transformers` is the only route to the requested variant. Verified:
  backbone depths `[2,2,2,2]`, 300 queries, contiguous 0-79 class ids.
- **EfficientDet-Lite0**: `effdet`, not TF Hub. **`tfhub.dev` now returns HTTP 404**, and
  the Kaggle Models path 404s as well. The `effdet` PyTorch port loads the same Google
  AutoML checkpoint and removes the TensorFlow dependency entirely.
  Note: `num_classes=90` (COCO-91 id space) and `image_size=320`, unlike the other
  models' 640. Both affect the comparisons in Q2 and Q3.
- **YOLO-NAS-S**: `super-gradients`, but its weight host `sghub.deci.ai` no longer
  resolves (Deci was acquired by NVIDIA and the hub was retired). The official checkpoint
  is still served from the CloudFront mirror referenced by super-gradients' master branch.
  `checkpoint_utils.py` derives its cache filename by splitting on the dead URL, so simply
  overriding `MODEL_URLS` raises `IndexError`. The working fix is to pre-seed torch's hub
  cache with the file under the expected name.

### Setup

```bash
cd Assignment2

python3.12 -m venv .venv       && ./.venv/bin/pip install -r requirements.txt
python3.10 -m venv .venv-nas   && ./.venv-nas/bin/pip install -r requirements-yolonas.txt
```

YOLO-NAS weights, since the official host is offline:

```bash
D=$(./.venv-nas/bin/python -c 'import torch;print(torch.hub.get_dir())')/checkpoints
mkdir -p "$D"
curl -L -o "$D/yolo_nas_s_coco.pth" \
  https://d2gjn4b69gu75n.cloudfront.net/models/yolo_nas_s_coco.pth
```

### Reproduction

Q1 runs from `model_preview.ipynb`. The sample image and the annotated output from all
models are written to `output/q1_<modelname>.jpg`, with the six panels combined in
`output/q1_all_models.jpg`.

Q4, the real-time demo:

```bash
curl -L -o data/demo.mp4 \
  https://github.com/intel-iot-devkit/sample-videos/raw/master/person-bicycle-car-detection.mp4

python run_demo.py --source data/demo.mp4 --model yolo11n.pt --conf 0.4 \
                   --output output/annotated_out.mp4
```

The 15 second submission clip is cut from the busy segment of the video. Re-encoding
rather than stream-copying keeps the cut frame-accurate:

```bash
ffmpeg -ss 34.3 -i output/annotated_out.mp4 -t 15 -c:v libx264 output/demo_clip.mp4

# or as a GIF
ffmpeg -ss 34.3 -t 15 -i output/annotated_out.mp4 \
       -vf "fps=10,scale=640:-1:flags=lanczos" output/demo_clip.gif
```

Q5 to Q12 run in `Assignment2.ipynb` on a Colab GPU runtime. `tensorrt` is deliberately
not pip-installed there: it replaces the cuDNN and NCCL wheels that PyTorch links against
and breaks training. Ultralytics auto-installs it during the first engine export, which
works.
