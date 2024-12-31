import os
import json
import random
import datetime
import torch
import torchvision.transforms as T
from torchvision import models
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np

# Paths
TRAIN_IMAGES_PATH = '/kaggle/input/coco-2017-dataset/coco2017/train2017'
VAL_IMAGES_PATH = '/kaggle/input/coco-2017-dataset/coco2017/val2017'
ANNOTATIONS_PATH = '/kaggle/input/coco-2017-dataset/coco2017/annotations'
ANNOTATIONS_FILE = f"{ANNOTATIONS_PATH}/instances_train2017.json"
WORKING_DIR = '/kaggle/working'

# Constants
CATEGORIES = ['person', 'cat', 'dog']
NUM_IMAGES_PER_CATEGORY = 1000
TRAIN_SPLIT = 0.8

# Load COCO annotations
with open(ANNOTATIONS_FILE, 'r') as f:
    coco_data = json.load(f)

# Filter images by category
category_to_images = {cat: [] for cat in CATEGORIES}
category_ids = {cat: next(c["id"] for c in coco_data["categories"] if c["name"] == cat) for cat in CATEGORIES}
id_to_category = {v: k for k, v in category_ids.items()}

# Create a mapping of image_id to annotations
image_to_annotations = {}
for ann in coco_data["annotations"]:
    cat_id = ann["category_id"]
    if cat_id in id_to_category:
        image_id = ann["image_id"]
        if image_id not in image_to_annotations:
            image_to_annotations[image_id] = []
        image_to_annotations[image_id].append(ann)
        category_to_images[id_to_category[cat_id]].append(image_id)

# Deduplicate and sample
for cat in CATEGORIES:
    category_to_images[cat] = list(set(category_to_images[cat]))[:NUM_IMAGES_PER_CATEGORY]

# Split into training and validation
train_images, val_images = [], []
for cat in CATEGORIES:
    random.shuffle(category_to_images[cat])
    split_idx = int(len(category_to_images[cat]) * TRAIN_SPLIT)
    train_images.extend(category_to_images[cat][:split_idx])
    val_images.extend(category_to_images[cat][split_idx:])

class COCOImageDataset(Dataset):
    def __init__(self, image_ids, images_path, annotations, image_to_annotations, category_ids):
        self.image_ids = image_ids
        self.images_path = images_path
        self.annotations = annotations
        self.image_to_annotations = image_to_annotations
        self.category_ids = category_ids
        self.cat_id_to_idx = {cat_id: idx + 1 for idx, cat_id in enumerate(category_ids.values())}

        self.transforms = T.Compose([
            T.Resize((600, 600)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        
        image_info = next(img for img in self.annotations['images'] if img['id'] == image_id)
        image_path = os.path.join(self.images_path, image_info['file_name'])
        image = Image.open(image_path).convert('RGB')

        image_anns = self.image_to_annotations.get(image_id, [])
        
        boxes = []
        labels = []
        
        for ann in image_anns:
            if ann['category_id'] in self.cat_id_to_idx:
                x, y, w, h = ann['bbox']
                boxes.append([x, y, x + w, y + h])
                labels.append(self.cat_id_to_idx[ann['category_id']])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([image_id])
        }

        image = self.transforms(image)
        return image, target

train_dataset = COCOImageDataset(train_images, TRAIN_IMAGES_PATH, coco_data, 
                                image_to_annotations, category_ids)
val_dataset = COCOImageDataset(val_images, TRAIN_IMAGES_PATH, coco_data, 
                              image_to_annotations, category_ids)

def collate_fn(batch):
    return tuple(zip(*batch))

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, 
                         num_workers=4, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, 
                       num_workers=4, collate_fn=collate_fn)

weights = models.detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
model = models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(CATEGORIES) + 1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, targets in loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()
        running_loss += losses.item()
    return running_loss / len(loader)

def save_and_display_plot(data, xlabel, ylabel, title, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    
    # Save the plot
    filepath = os.path.join(WORKING_DIR, f"{filename}_{timestamp}.png")
    plt.savefig(filepath)
    
    # Display the plot
    plt.show()
    
    print(f"Saved plot: {filepath}")

def save_and_display_roc_curve(fpr, tpr, roc_auc):
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid()
    
    # Save the plot
    filepath = os.path.join(WORKING_DIR, f"roc_curve_{timestamp}.png")
    plt.savefig(filepath)
    
    # Display the plot
    plt.show()
    
    print(f"Saved plot: {filepath}")

def evaluate_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in loader:
            images = list(image.to(device) for image in images)
            outputs = model(images)
            for i, output in enumerate(outputs):
                true_labels = targets[i]['labels'].cpu().numpy()
                predicted_labels = output['labels'].cpu().numpy()
                correct += np.isin(predicted_labels, true_labels).sum()
                total += len(true_labels)
    return correct / total if total > 0 else 0

def compute_mAP_and_ROC(model, loader, device):
    all_labels = []
    all_scores = []
    precisions = []
    recalls = []
    
    for images, targets in loader:
        images = list(image.to(device) for image in images)
        outputs = model(images)
        for i, output in enumerate(outputs):
            scores = output['scores'].detach().cpu().numpy()
            predicted_labels = output['labels'].detach().cpu().numpy()
            true_labels = targets[i]['labels'].cpu().numpy()
            all_scores.extend(scores)
            all_labels.extend(np.isin(predicted_labels, true_labels).astype(int))
            if len(predicted_labels) > 0:
                precisions.append(len(set(predicted_labels) & set(true_labels)) / len(predicted_labels))
            else:
                precisions.append(0.0)
            if len(true_labels) > 0:
                recalls.append(len(set(predicted_labels) & set(true_labels)) / len(true_labels))
            else:
                recalls.append(0.0)
    
    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    roc_auc = auc(fpr, tpr)
    
    return precisions, recalls, fpr, tpr, roc_auc

num_epochs = 30
losses = []
accuracies = []

for epoch in range(num_epochs):
    loss = train_one_epoch(model, train_loader, optimizer, device)
    losses.append(loss)
    accuracy = evaluate_accuracy(model, val_loader, device)
    accuracies.append(accuracy)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

model_path = os.path.join(WORKING_DIR, f"model_{timestamp}.pth")
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")


save_and_display_plot(losses, 'Epoch', 'Loss', 'Training Loss', 'loss_plot')
save_and_display_plot(accuracies, 'Epoch', 'Accuracy', 'Validation Accuracy', 'accuracy_plot')

precisions, recalls, fpr, tpr, roc_auc = compute_mAP_and_ROC(model, val_loader, device)
save_and_display_roc_curve(fpr, tpr, roc_auc)
save_and_display_plot(precisions, 'Recall', 'Precision', 'Precision-Recall Curve', 'pr_curve')
