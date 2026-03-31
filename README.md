# Quant-Research: CIFARQuant

Quantization analysis of a convolutional neural network on **CIFAR-10**, comparing **FP32**, **Post-Training Quantization (PTQ)**, and **Quantization-Aware Training (QAT)** across model accuracy, model size, and CPU inference time.

---

## Overview

This project evaluates how quantization affects CNN performance on the CIFAR-10 dataset. The goal is to compare the tradeoffs between:

- **FP32 baseline**
- **PTQ (INT8)**
- **QAT (INT8)**

Each model is evaluated using:

- **Test accuracy**
- **Model size**
- **Average CPU inference time per batch**

This repository focuses on whether quantization can reduce memory usage and improve inference speed while preserving as much accuracy as possible.

---

## Methods

### FP32 Baseline
A pretrained CNN was fine-tuned and evaluated in full precision.

### Post-Training Quantization (PTQ)
The fine-tuned FP32 model was converted to **INT8** using post-training quantization. This approach improves efficiency without retraining under quantization.

### Quantization-Aware Training (QAT)
The pretrained FP32 model was fine-tuned for additional epochs using quantization-aware training, then converted to **INT8**. This allows the model to adapt to quantization effects during training.

---

## Results

| Model           | Accuracy (%) | Size (MB) | Inference Time (s/batch) |
|----------------|-------------:|----------:|--------------------------:|
| FP32 fine-tuned |        80.88 |      2.38 |                  0.005061 |
| PTQ (INT8)      |        80.53 |      0.62 |                  0.001983 |
| QAT (INT8)      |        80.64 |      0.62 |                  0.001931 |

---

## Summary

The results show that both quantized models significantly reduced model size and improved inference speed compared to the FP32 baseline.

- **FP32** achieved the highest accuracy overall at **80.88%**
- **PTQ** reduced model size from **2.38 MB** to **0.62 MB** and improved inference speed substantially, with only a small drop in accuracy
- **QAT** achieved slightly better accuracy than PTQ while keeping the same compact model size and nearly identical inference speed

Overall, **QAT provided the best quantized accuracy**, while **PTQ provided a simpler quantization approach with similar efficiency gains**. Both INT8 methods reduced model size by about **74%** and improved CPU inference speed by roughly **2.5x** relative to FP32.

---

## Key Takeaways

- Quantization greatly reduced memory usage
- INT8 inference was much faster than FP32 on CPU
- QAT preserved accuracy slightly better than PTQ
- FP32 still produced the best overall accuracy

---

## Repository Goals

This project was created to explore practical tradeoffs in neural network quantization, including:

- efficiency vs. accuracy
- model compression
- CPU deployment performance
- comparison of PTQ and QAT workflows

---

## Future Work

Possible next steps include:

- testing additional CNN architectures
- expanding to other datasets
- comparing different quantization backends
- adding latency measurements on different hardware
- improving QAT with layer fusion support

---

## Dataset

This project uses the **CIFAR-10** dataset, which contains 60,000 32x32 color images across 10 classes.

---

## Notes

For fair comparison, the FP32 baseline was fine-tuned for the same number of additional epochs used in the QAT experiment before comparing final results.

---