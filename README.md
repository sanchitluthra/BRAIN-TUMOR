# 🧠 Brain Tumor Analysis — Segmentation & Classification

A deep learning pipeline for brain tumor **segmentation** and **classification** from MRI scans. Simple design, custom loss, strong results.

---

## 🔬 Task 1 — Segmentation (BRISC 2025)

**Model:** MIT-b5 encoder + U-Net decoder with scSE Attention  
**Input:** Complete MRI scans (512×512) — no cropping, full images fed directly to the model  
**Optimizer:** AdamW + Warmup Cosine LR + SWA  
**Post-processing:** 8-aug D4 TTA + Threshold Search (best threshold: 0.5)  
**Dice Score: 90.33%**

### Custom Loss Function

```
Loss = 0.5 × Dice + 0.3 × FocalTversky + 0.2 × LabelSmoothBCE
```

- **Dice Loss** — optimizes overlap between prediction and ground truth
- **Focal Tversky Loss** — penalizes missed tumor pixels more than false alarms, critical in medical imaging
- **Label Smooth BCE** — prevents overconfidence on fuzzy tumor boundaries

### IoU vs Swin-HAFNet (official BRISC 2025 benchmark)

Swin-HAFNet is the transformer model proposed in the original BRISC 2025 paper as the official dataset benchmark.

| Method | IoU |
|--------|-----|
| Swin-HAFNet *(BRISC 2025 paper)* | 82.30% |
| **Ours** *(SWA + 8-aug TTA)* | **83.14%** |

> Our model surpasses the official BRISC benchmark with a simpler 2D approach.

---

## 🔍 Task 2 — Classification

**Model:** ResNet50 (pretrained, fine-tuned)  
**Input:** Cropped tumor regions (384×384) — region of interest extracted before classification  
**Classes:** 4 tumor categories (Glioma, Meningioma, Pituitary, No Tumor)  
**Training:** Two-stage — warmup (head only) → fine-tune (deeper layers)  
**Optimizer:** AdamW + Cosine Annealing + Early Stopping  

### Results

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Glioma | 99% | 98% | 99% |
| Meningioma | 98% | 98% | 98% |
| No Tumor | 99% | 100% | 99% |
| Pituitary | 100% | 99% | 99% |
| **Overall Accuracy** | | | **98.9%** |

---

## ⚙️ Key Design Choices

- **Full MRI for segmentation** — no cropping or ROI extraction; complete scans are passed directly to the model
- **Cropped input for classification** — tumor region is extracted before feeding into the classifier
- **MIT-b5 encoder** — captures both local detail and global context without the overhead of a full Swin backbone
- **scSE attention** — spatial and channel squeeze-excitation in the decoder for better feature recalibration
- **SWA** — stochastic weight averaging from epoch 50 onward for better generalization
- **Custom combined loss** — designed specifically for the challenges of medical image segmentation
  
