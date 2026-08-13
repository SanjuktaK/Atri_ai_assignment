# Assignment 1

The report can be found under 
Assignment1/Assignment1.ipynb
and the [wandb report](https://api.wandb.ai/links/trishab/8xad8n5m)



# Assignment 2 - Investigating Small Vision Models for Efficient Edge AI Deployment

Applied ML research assignment comparing five lightweight object detectors for real-time, resource-constrained edge deployment.

### Environments

Two environments were used. The split is not a preference, it is forced by a hard version conflict, documented below.

| Environment | Python | Used for | Key packages |
|---|---|---|---|
| `.venv` (local) | 3.12.11 | Q1 inference, Q3 CPU/MPS benchmarks, Q4 demo | torch 2.13.0, torchvision 0.28.0, numpy 2.5.2, ultralytics 8.4.117, transformers 5.15.0, timm 1.0.28, effdet 0.4.1, onnx 1.22.0, onnxruntime 1.28.0, opencv-python 5.0.0.93, pycocotools 2.0.11 |
| `.venv-nas` (local) | 3.10.19 | YOLO-NAS-S only | super-gradients 3.7.1, torch 2.1.2, torchvision 0.16.2, numpy 1.23.0, onnx 1.15.0, onnxruntime 1.15.0, protobuf 3.20.3, torchmetrics 0.8.0, opencv-python 4.11.0.86 |

Q1, Q3 and Q4 were run locally (Apple Silicon, CPU/MPS). Q5–Q10 and Q12 require CUDA and were run on a Colab A100; all latency and FPS figures in those sections are A100 measurements and are not comparable to the local numbers in Q3.

### Why two local environments

`super-gradients` 3.7.1 (the latest release, unchanged since 2024) pins `numpy<=1.23`,
while `ultralytics` requires `numpy>=1.23.0`. The two intersect at exactly one version,
`numpy==1.23.0`, which ships no wheels beyond CPython 3.10. Any environment containing
super-gradients is therefore pinned to Python 3.10 and, transitively, to
`onnx==1.15.0`, `onnxruntime==1.15.0`, `protobuf==3.20.3` and `torchmetrics==0.8` —
all mid-2023 releases. `onnxruntime` no longer publishes cp310 wheels at all.

Admitting super-gradients to the main environment would have frozen the entire project
at that stack, including the ONNX Runtime and INT8 work in Q5 and the pruning in Q10.
Isolating it costs one extra `pip install` in the README and keeps everything else current.
YOLO-NAS-S is invoked from the main notebook as a subprocess (`yolo_nas_infer.py`),
so the analysis stays in one place.
The Ultralytics YOLO-NAS integration was evaluated as an alternative to a second environment, but ultralytics/models/nas/model.py imports super_gradients unconditionally, so it inherits the same numpy<=1.23 constraint. It offers a different API to the same dependency, not a way around it.

### Model sources, and why they differ from the brief

- **YOLOv8n, YOLO11n, YOLO26n** — `ultralytics`, as suggested.
- **RT-DETR (ResNet-18)** — `transformers` (`PekingU/rtdetr_r18vd`). The Ultralytics
  RT-DETR wrapper ships only `rtdetr-l` (ResNet-50) and `rtdetr-x` (ResNet-101); there is
  no R18 checkpoint. `transformers` is the only route to the requested variant. Verified:
  backbone depths `[2,2,2,2]`, 300 queries, contiguous 0–79 class ids.
- **EfficientDet-Lite0** — `effdet`, not TF Hub. **`tfhub.dev` now returns HTTP 404**;
  the Kaggle Models path 404s as well. The `effdet` PyTorch port loads the same Google
  AutoML checkpoint and removes the TensorFlow dependency entirely.
  Note: `num_classes=90` (COCO-91 id space) and `image_size=320`, unlike the other
  models' 640, both affect the comparisons in Q2 and Q3.
- **YOLO-NAS-S** — `super-gradients`, but its weight host `sghub.deci.  ai` no longer resolves (Deci was acquired by NVIDIA and the hub was    retired). The official checkpoint is still served from the CloudFront mirror referenced by super-gradients' master branch.
  `checkpoint_utils.py` derives its cache filename by splitting on the dead URL, so simply
  overriding `MODEL_URLS` raises `IndexError`; the working fix is to pre-seed torch's hub
  cache with the file under the expected name.

### Reproduction

python3.12 -m venv .venv       && ./.venv/bin/pip install -r requirements.txt
python3.10 -m venv .venv-nas   && ./.venv-nas/bin/pip install -r requirements-yolonas.txt

# YOLO-NAS weights (official host is offline)
D=$(./.venv-nas/bin/python -c 'import torch;print(torch.hub.get_dir())')/checkpoints
mkdir -p "$D" && curl -L -o "$D/yolo_nas_s_coco.pth" \
  https://d2gjn4b69gu75n.cloudfront.net/models/yolo_nas_s_coco.pth


The sample image with output from 5 different models are present in the report Assignment2/model_preview.ipynb
With all the output images from every model stored at output/q1_'modelname'.jpg


curl -L -o data/demo.mp4 \
  https://github.com/intel-iot-devkit/sample-videos/raw/master/person-bicycle-car-detection.mp4

python run_demo.py --source data/demo.mp4 --model yolo11n.pt --conf 0.4 \
                   --output output/annotated_out.mp4

# 15s submission clip from the busy segment (re-encode, so the cut is frame-accurate)
ffmpeg -ss 34.3 -i output/annotated_out.mp4 -t 15 -c:v libx264 output/demo_clip.mp4

# or as a GIF
ffmpeg -ss 34.3 -t 15 -i output/annotated_out.mp4 \
       -vf "fps=10,scale=640:-1:flags=lanczos" output/demo_clip.gif
