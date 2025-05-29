This project uses the [DINOv2](https://github.com/facebookresearch/dinov2) vision transformer model to perform unsupervised anomaly detection on dynamic dataset for defect detection. It compares test images against good samples using feature similarity to detect defects.

## How It Works

1. **Extract features** from good (non-defective) training images using DINOv2.
2. For each test image:
   - Compute its embedding.
   - Measure **cosine similarity** to all training embeddings.
   - Invert the maximum similarity to get an **anomaly score**.
3. Optionally visualize:
   - Per-image anomaly scores
   - Class-wise boxplot of anomaly distributions.

## Model

- Backbone: `dinov2_vits14` from Facebook's DINOv2 repo
- Feature: Global average pooled `x_norm_patchtokens`

## Dataset Structure

```

bottle/
├── train/
│   └── good/
├── test/
│   ├── good/
│   ├── broken\_large/
│   ├── broken\_small/
│   └── contamination/
└── ground\_truth/

```

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/bottle-anomaly-detection.git
cd bottle-anomaly-detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the script

```bash
python dinov2_bottle_anomaly.py
```

## Optional Visualizations

Uncomment the following lines in the script if needed:

* Boxplot of anomaly scores by defect type:

```python
visualize_anomaly_scores(scores_by_class)
```

* Individual image display with anomaly score:

```python
visualize(img_path, score)
```

## Output

* Anomaly score for each test image.
* Optional matplotlib visualizations:

  * Class-level anomaly distribution
  * Image with score title

## Dataset test link
- https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads

