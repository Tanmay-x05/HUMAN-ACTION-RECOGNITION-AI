#  Human Action Recognition AI
### Skeleton-based action detection using Spatial-Temporal Graph Convolutional Networks

> *Teaching machines to understand human movement — one joint at a time.*

---

##  What is this?

A deep learning system that **watches how your skeleton moves** and figures out what you're doing — sitting, falling, waving, kicking, jumping, and 56 other actions. No pixels. No RGB. Just bones.

Built on **ST-GCN (Spatial-Temporal Graph Convolutional Networks)**, this project treats the human body as a **graph** — 17 joints, 19 anatomical edges — and learns the spatiotemporal patterns that distinguish one action from another.

**~85–87% accuracy on 60 action classes. Baseline HOG+SVM? 52%. Two-stream CNN? 74%.** We win.

---

##  How it Works

```
Raw Skeleton Data → Preprocessing → ST-GCN (3 blocks) → Action Label
```

-  **17 body joints** modelled as a spatial graph (COCO format)
-  **50 frames** per sequence, uniformly resampled
-  **6 input channels** — x, y coordinates + velocity + acceleration (first & second order temporal differences)
-  **3 stacked ST-GCN blocks** (GCN + TCN with residual connections)
-  **60 action classes** — from drinking water to falling down to jumping

---

##  Dataset

**NTU RGB+D 60** — 56,880 video samples, 60 action classes, skeleton keypoints extracted via HRNet.

>  **Download here:** [NTU RGB+D 60 Skeleton Data on Kaggle](https://www.kaggle.com/datasets/hungkhoi/skeleton-data-of-ntu-rgbd-60-dataset)

Place the downloaded file at:
```
data/ntu60_hrnet.pkl
```

Split used:
| Split      | Samples   |
|------------|-----------|
| Train (70%)| ~39,816   |
| Val (15%)  | ~8,532    |
| Test (15%) | ~8,532    |

---

##  Architecture

| Layer           | Input Shape        | Output Shape       |
|-----------------|--------------------|--------------------|
| BatchNorm1D     | (B, 6×17, T)       | (B, 102, T)        |
| ST-GCN Block 1  | (B, 6, T, 17)      | (B, 64, T, 17)     |
| ST-GCN Block 2  | (B, 64, T, 17)     | (B, 128, T, 17)    |
| ST-GCN Block 3  | (B, 128, T, 17)    | (B, 256, T, 17)    |
| Global Avg Pool | (B, 256, T, 17)    | (B, 256)           |
| Fully Connected | (B, 256)           | (B, 60)            |

---

##  Training Details

| Hyperparameter     | Value                          |
|--------------------|-------------------------------|
| Optimizer          | Adam (lr=1e-3, β1=0.9, β2=0.999) |
| LR Scheduler       | Cosine Annealing (T_max=80)   |
| Loss               | CrossEntropyLoss               |
| Batch Size         | 32                             |
| Max Epochs         | 80 (early stopping, patience=15) |
| Dropout            | 0.4 (in each TCN block)        |
| Augmentation       | Gaussian noise (σ=0.01) + Random H-flip (p=0.5) |
| Class Balancing    | WeightedRandomSampler          |
| Random Seed        | 42                             |

---

##  Results

| Model                          | Accuracy |
|-------------------------------|----------|
| Random Baseline                | 1.67%    |
| HOG + SVM                      | ~52.4%   |
| Two-Stream CNN (RGB + Optical Flow) | ~73.6%   |
| ST-GCN (no augmentation)       | ~81.2%   |
| **ST-GCN (ours, full pipeline)** | **~85.7%** |

---

##  Quickstart

```bash
# Clone the repo
git clone https://github.com/your-username/human-action-recognition
cd human-action-recognition

# Install dependencies
pip install torch numpy scikit-learn tqdm

# Download dataset from Kaggle and place at:
# data/ntu60_hrnet.pkl

# Train
python train.py

# Evaluate
python evaluate.py
```

---

##  Future Work

- 🤖 Transformer-based temporal encoding for long-range dependencies
- 📈 Scale to NTU RGB+D 120 (120 action classes)
- 👥 Multi-person graph modelling
- ⚡ Real-time deployment via TensorRT / ONNX — for fall detection, gesture interfaces, and more

---

## 📄 References

- Yan et al., *Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition*, AAAI 2018
- Simonyan & Zisserman, *Two-Stream Convolutional Networks*, NIPS 2014
- Wang et al., *Temporal Segment Networks*, IEEE TPAMI 2019

---

