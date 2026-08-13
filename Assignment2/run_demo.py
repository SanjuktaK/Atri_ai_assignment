import argparse
import time
from collections import deque

import cv2
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("--source", required=True, help="video path or 0 for webcam")
parser.add_argument("--model", default="yolo11n.pt")
parser.add_argument("--conf", type=float, default=0.4)
parser.add_argument("--output", default="annotated_out.mp4")
args = parser.parse_args()

source = int(args.source) if args.source.isdigit() else args.source
cap = cv2.VideoCapture(source)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_in = cap.get(cv2.CAP_PROP_FPS) or 30    # webcams report 0

writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (width, height))
model = YOLO(args.model)

frame_times = deque(maxlen=30)   # rolling window for the FPS counter
latencies = []
frames = 0
start = time.perf_counter()

while True:
    loop_start = time.perf_counter()
    ok, frame = cap.read()
    if not ok:
        break

    t0 = time.perf_counter()
    result = model(frame, conf=args.conf, verbose=False)[0]
    latencies.append((time.perf_counter() - t0) * 1000)

    annotated = result.plot()    # draws boxes, class labels and confidence
    frame_times.append(time.perf_counter() - loop_start)

    fps = len(frame_times) / sum(frame_times)
    cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    writer.write(annotated)
    frames += 1

cap.release()
writer.release()

elapsed = time.perf_counter() - start
print("total frames processed:", frames)
print(f"average FPS: {frames / elapsed:.2f}")
print(f"average inference latency: {sum(latencies) / len(latencies):.2f} ms")
