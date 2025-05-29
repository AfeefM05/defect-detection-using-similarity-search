import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets.folder import default_loader
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from collections import defaultdict

# Load DINOv2
print("Loading DINOv2 model...")
dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').cuda().eval()

# Directories
root_dir = "wood"
train_dir = os.path.join(root_dir, 'train', 'good')
test_dir = os.path.join(root_dir, 'test')
ground_truth_dir = os.path.join(root_dir, 'ground_truth')

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)  # Normalize for DINOv2
])

# Load image embedding
def get_embedding(img_path):
    img = default_loader(img_path)
    img = transform(img).unsqueeze(0).cuda()
    with torch.no_grad():
        emb = dinov2_model.forward_features(img)["x_norm_patchtokens"]
        emb = emb.mean(dim=1)  # Global average pooling
    return emb.cpu()

print("Extracting training embeddings...")
train_embeddings = []
for fname in tqdm(os.listdir(train_dir)):
    fpath = os.path.join(train_dir, fname)
    emb = get_embedding(fpath)
    train_embeddings.append(emb)
train_embeddings = torch.cat(train_embeddings, dim=0)

# Anomaly score

def anomaly_score(test_embedding, train_embeddings):
    sims = F.cosine_similarity(test_embedding, train_embeddings)
    max_sim, _ = torch.max(sims, dim=0)
    return 1 - max_sim.item()  # Higher => more anomalous

# Score storage for plotting
scores_by_class = defaultdict(list)

def visualize(img_path, score):
    img = Image.open(img_path).resize((224, 224))
    plt.imshow(img)
    plt.title(f"Anomaly Score: {score:.4f}")
    plt.axis("off")
    plt.show()
    
print("\nEvaluating on test set and collecting scores...")
for subdir in sorted(os.listdir(test_dir)):
    subdir_path = os.path.join(test_dir, subdir)
    for fname in tqdm(os.listdir(subdir_path), desc=f"Processing {subdir}"):
        fpath = os.path.join(subdir_path, fname)
        test_emb = get_embedding(fpath)
        score = anomaly_score(test_emb, train_embeddings)
        # visualize(fpath,score)
        scores_by_class[subdir].append(score)

def visualize_anomaly_scores(scores_by_class):
    plt.figure(figsize=(12, 6))
    labels = list(scores_by_class.keys())
    values = [scores_by_class[k] for k in labels]
    plt.boxplot(values, labels=labels)
    plt.ylabel("Anomaly Score")
    plt.title("Anomaly Score Distribution per Defect Type")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

visualize_anomaly_scores(scores_by_class)