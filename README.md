# Quant-Research: CIFARQuant
Quantization analysis of CNN models on CIFAR dataset, comparing FP32, PTQ, and QAT across accuracy, size, and performance.
---
## Post-Training Quantization (PTQ) Results
A CNN was trained on CIFAR-10 and evaluated before and after applying post-training quantization (INT8).
### Results
| Model       | Accuracy (%) | Size (MB) | Inference Time (s/batch) |
|-------------|-------------:|----------:|--------------------------:|
| FP32        |        79.67 |      2.37 |                  0.00487 |
| PTQ (INT8)  |        79.52 |      0.61 |                  0.00187 |
### Summary
- Accuracy drop is minimal (-0.15%), indicating strong quantization robustness
- Model size reduced by ~74%
- Inference speed improved by ~2.6× on CPU
- Post-training quantization provides significant efficiency gains (model size and inference speed) with negligible accuracy loss, making it a practical optimization for deployment on resource-constrained systems.