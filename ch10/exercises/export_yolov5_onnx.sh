#!/usr/bin/env bash
# Export the pretrained YOLOv5s weights to ONNX (ch10/exercises/yolov5s.onnx) so that the C++
# exercise can run them with OpenCV's DNN module.
#
# Requirements: a Python environment with PyTorch, e.g. the `pora` conda environment of the
# original Python exercises (conda activate pora). The export needs opset 12 and graph
# simplification for OpenCV 4.5 to be able to load the model.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="${HERE}/.yolov5_export"
mkdir -p "$WORK"

# Use the weights shipped with the Python exercises if available, otherwise download them.
WEIGHTS="$WORK/yolov5s.pt"
PY_WEIGHTS="${HERE}/../../../pora-exercises/ch10/exercises/yolov5s.pt"
if [ ! -f "$WEIGHTS" ]; then
  if [ -f "$PY_WEIGHTS" ]; then
    cp "$PY_WEIGHTS" "$WEIGHTS"
  else
    curl -L -o "$WEIGHTS" https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt
  fi
fi

[ -d "$WORK/yolov5" ] || git clone --depth 1 https://github.com/ultralytics/yolov5 "$WORK/yolov5"
pip install -q -r "$WORK/yolov5/requirements.txt" onnx onnxslim
python "$WORK/yolov5/export.py" --weights "$WEIGHTS" --include onnx --opset 12 --simplify
mv "${WEIGHTS%.pt}.onnx" "$HERE/yolov5s.onnx"
echo "Saved $HERE/yolov5s.onnx"
