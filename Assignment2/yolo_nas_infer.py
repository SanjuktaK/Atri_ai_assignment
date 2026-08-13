"""
YOLO-NAS-S inference. 
Runs in .venv-nas because super-gradients pins numpy<=1.23.
"""
import argparse
import json
import sys
import time
import cv2
import numpy as np
from super_gradients.training import models

parser = argparse.ArgumentParser()
parser.add_argument("--source", default="data/sample.jpg")
parser.add_argument("--out", default="output/q1_yolo_nas_s.jpg")
parser.add_argument("--summary", default="output/q1_yolo_nas_s.json")
parser.add_argument("--conf", type=float, default=0.4)
parser.add_argument("--benchmark", action="store_true")
parser.add_argument("--warmup", type=int, default=10)
parser.add_argument("--iters", type=int, default=100)

args = parser.parse_args()

model = models.get("yolo_nas_s", pretrained_weights="coco")
model.eval()

model.predict(args.source, conf=args.conf)                  # warm-up
start = time.perf_counter()
result = model.predict(args.source, conf=args.conf)
latency_ms = (time.perf_counter() - start) * 1000

pred = result.prediction
summary = {
    "labels": [result.class_names[int(c)] for c in pred.labels],
    "scores": [round(float(s), 3) for s in pred.confidence],
    "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
    "size_mb": round(sum(p.numel() for p in model.parameters()) * 4 / 1e6, 2),
}


if args.benchmark:
    # preloaded RGB array, matching how the main-env models are timed
    img = cv2.cvtColor(cv2.imread(args.source), cv2.COLOR_BGR2RGB)

    for _ in range(args.warmup):
        model.predict(img, conf=args.conf, fuse_model=False)

    times = []
    for _ in range(args.iters):
        start = time.perf_counter()
        model.predict(img, conf=args.conf, fuse_model=False)   # skip re-fusing each call
        times.append((time.perf_counter() - start) * 1000)

    summary.update(
        device="cpu",
        iters=args.iters,
        latency_ms=round(float(np.mean(times)), 2),
        std_ms=round(float(np.std(times)), 2),
        fps=round(1000 / float(np.mean(times)), 2),
    )

with open(args.summary, "w") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary), file=sys.stderr)   # super-gradients hijacks stdout


