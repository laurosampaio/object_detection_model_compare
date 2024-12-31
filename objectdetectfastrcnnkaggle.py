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

# Paths
TRAIN_IMAGES_PATH = '/kaggle/input/coco-2017-dataset/coco2017/train2017'
VAL_IMAGES_PATH = '/kaggle/input/coco-2017-dataset/coco2017/val2017'
ANNOTATIONS_PATH = '/kaggle/input/coco-2017-dataset/coco2017/annotations'
ANNOTATIONS_FILE = f"{ANNOTATIONS_PATH}/instances_train2017.json"
OUTPUT_DIR = '/kaggle/working'

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
    if cat_id in id_to_category:  # Only include annotations for our categories
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
        
        # Get image info and load image
        image_info = next(img for img in self.annotations['images'] if img['id'] == image_id)
        image_path = os.path.join(self.images_path, image_info['file_name'])
        image = Image.open(image_path).convert('RGB')
        
        # Get annotations for this image
        image_anns = self.image_to_annotations.get(image_id, [])
        
        # Prepare boxes and labels
        boxes = []
        labels = []
        
        for ann in image_anns:
            if ann['category_id'] in self.cat_id_to_idx:
                # COCO bbox format is [x, y, width, height]
                # Convert to [x1, y1, x2, y2] format
                x, y, w, h = ann['bbox']
                boxes.append([x, y, x + w, y + h])
                labels.append(self.cat_id_to_idx[ann['category_id']])
        
        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Create target dictionary
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([image_id])
        }
        
        image = self.transforms(image)
        return image, target

# Create datasets
train_dataset = COCOImageDataset(train_images, TRAIN_IMAGES_PATH, coco_data, 
                                image_to_annotations, category_ids)
val_dataset = COCOImageDataset(val_images, VAL_IMAGES_PATH, coco_data, 
                              image_to_annotations, category_ids)

# DataLoaders with collate_fn to handle variable size inputs
def collate_fn(batch):
    return tuple(zip(*batch))

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, 
                         num_workers=4, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, 
                       num_workers=4, collate_fn=collate_fn)

# Model setup
weights = models.detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
model = models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)

# Update the classifier head for our categories (+1 for background)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(CATEGORIES) + 1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Loss and optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

# Training function
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

# Training loop
num_epochs = 5
losses = []
for epoch in range(num_epochs):
    loss = train_one_epoch(model, train_loader, optimizer, device)
    losses.append(loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}")

# Save model and loss plot
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = os.path.join(OUTPUT_DIR, f"model_{timestamp}.pth")
plot_path = os.path.join(OUTPUT_DIR, f"loss_plot_{timestamp}.png")

torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

plt.plot(range(1, num_epochs + 1), losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid()
plt.savefig(plot_path)
plt.show()

print(f"Model saved to {model_path}")
print(f"Loss plot saved to {plot_path}")