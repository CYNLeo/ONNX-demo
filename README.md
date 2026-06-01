# ONNX Demo: YOLO Model Export and Inference

## Overview

This project demonstrates the deployment workflow of a YOLO object detection model using ONNX.

The project covers:

* Exporting a trained PyTorch model to ONNX
* Validating the exported model
* Running inference with ONNX Runtime
* Comparing deployment-ready model formats

The objective is to understand the model deployment pipeline commonly used in production AI systems.

---

## Project Structure

```text
ONNX-demo/
├── export_onnx.py
├── run_onnx.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Environment

* Python 3.10
* PyTorch
* Ultralytics YOLO
* ONNX
* ONNX Runtime

---

## Exporting to ONNX

The model is exported from PyTorch YOLO model to ONNX format.

Example:

```bash
python export_onnx.py
```

---

## Running Inference

Example:

```bash
python run_onnx.py
```

The inference pipeline:

1. Load ONNX model
2. Preprocess image
3. Execute ONNX Runtime session
4. Post-process detections
5. Visualize results

---

## Results

The exported ONNX model produces object detection outputs comparable to the original PyTorch model while enabling framework-independent deployment.

---

## Notes

Model weight files (.pt, .onnx) are not included in this repository due to size constraints.

Users can generate the ONNX model by running the export script on their own trained weights.

---


