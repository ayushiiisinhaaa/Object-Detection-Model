# Model Card

## Current model

The application currently loads the official `yolov8n.pt` checkpoint pretrained
on COCO. The Vercel runtime uses an ONNX export of YOLOv8n for compatibility with
serverless function limits. It is a deployment baseline, not a repository-trained model.

## Intended use

- Demonstrating an end-to-end object-detection application and API.
- Detecting the 80 object categories represented in COCO.
- Providing a baseline for a future custom wildlife and safety detector.

## Out-of-scope use

- Safety-critical monitoring or autonomous decisions.
- Identifying people or inferring sensitive personal attributes.
- Detecting custom wildlife or mining classes without validated custom weights.

## Evaluation

Repository-specific evaluation is pending because a versioned held-out dataset
and custom checkpoint are not yet available. Run `evaluate.py` to produce
`metrics.json`, PR curves, a confusion matrix, and validation samples once those
artifacts exist. Results must include the dataset version, split, thresholds,
hardware, and checkpoint checksum.

## Limitations

Accuracy may degrade for small, occluded, blurred, or out-of-distribution objects.
COCO categories and imagery do not represent every deployment environment. Model
outputs require human review for consequential use.
