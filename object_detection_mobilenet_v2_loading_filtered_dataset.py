#Imports
import os
import pandas as pd
import random
from pycocotools.coco import COCO
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc , precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import cv2
import numpy as np
import time
from datetime import datetime, timedelta

# Paths for the dataset, anotations, working directory and images

TRAIN_PATH = '/kaggle/input/coco-2017-dataset/coco2017/train2017'
VAL_PATH = '/kaggle/input/coco-2017-dataset/coco2017/val2017'
ANNOTATIONS_PATH = '/kaggle/input/coco-2017-dataset/coco2017/annotations'
FILTERED_DATASET = '/kaggle/input/filtered-coco-dataset'
WORKING_DIR = '/kaggle/working'
FILTERED_CATEGORIES = ['person', 'cat', 'dog']

# Load COCO annotations
annotations_file = os.path.join(ANNOTATIONS_PATH, 'instances_train2017.json')
coco = COCO(annotations_file)

# Get category IDs for the selected categories
category_ids = coco.getCatIds(catNms=FILTERED_CATEGORIES)

# Load split datasets
train_csv_path = os.path.join(FILTERED_DATASET, 'train_data.csv')
test_csv_path = os.path.join(FILTERED_DATASET, 'test_data.csv')

train_data = pd.read_csv(train_csv_path)
test_data = pd.read_csv(test_csv_path)

# Preprocess images
def preprocess_image(image_path, target_size=(224, 224)):
    """Load and preprocess an image for MobileNetV2."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = image / 255.0  # Normalize to [0, 1]
    return image

# Load images and labels
def load_data(data, path_prefix, target_size=(224, 224)):
    images = []
    labels = []
    for _, row in data.iterrows():
        image_path = os.path.join(path_prefix, row['image'])
        images.append(preprocess_image(image_path, target_size))
        labels.append(row['category_id'])
    return np.array(images), np.array(labels)

X_train, y_train = load_data(train_data, TRAIN_PATH)
X_test, y_test = load_data(test_data, TRAIN_PATH)

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = to_categorical(label_encoder.fit_transform(y_train))
y_test_encoded = to_categorical(label_encoder.transform(y_test))

# Build the model using MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Unfreeze some of the later layers for fine-tuning
for layer in base_model.layers[-30:]:  # Unfreeze last 30 layers
    layer.trainable = True

# Add improved classification head
inputs = Input(shape=(224, 224, 3))
x = base_model(inputs)
x = GlobalAveragePooling2D()(x)  # Use GAP instead of Flatten
x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
x = Dropout(0.5)(x)  # Add dropout for regularization
x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
x = Dropout(0.3)(x)
outputs = Dense(3, activation='softmax')(x)
model = Model(inputs, outputs)

# Compile with a lower learning rate
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer,
             loss='categorical_crossentropy',
             metrics=['accuracy'])

model.summary()

# Get current datetime and format it
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Define file path for the model summary
model_summary_path = os.path.join(WORKING_DIR, f'model_summary_{timestamp}.png')

# Save the model summary as an image
plot_model(model, to_file=model_summary_path, show_shapes=True, show_layer_names=True)

print(f"Model summary saved as an image at {model_summary_path}")

# Train the model
history = model.fit(
    X_train, y_train_encoded,
    validation_data=(X_test, y_test_encoded),
    epochs=10,  
    batch_size=32,
    verbose=1
)

# Save the trained model
model_path = os.path.join(WORKING_DIR, 'mobilenet_v2_coco.keras')
model.save(model_path)

print(f"Trained model saved at {model_path}")

# Get current datetime and format it
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Define file paths with timestamp
accuracy_loss_plot_path = os.path.join(WORKING_DIR, f'accuracy_loss_plot_{timestamp}.png')
roc_curve_plot_path = os.path.join(WORKING_DIR, f'roc_curve_plot_{timestamp}.png')

# Plot training accuracy and loss
plt.figure(figsize=(12, 6))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
# Save accuracy and loss plot with timestamp
plt.savefig(accuracy_loss_plot_path, dpi=300)  # Save with high resolution
plt.show()

print(f"Accuracy and loss plot saved at {accuracy_loss_plot_path}")

# ROC Curve
y_test_pred = model.predict(X_test)
fpr, tpr, _ = roc_curve(y_test_encoded.ravel(), y_test_pred.ravel())
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title('Receiver Operating Characteristic')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')

# Save ROC curve plot with timestamp
plt.savefig(roc_curve_plot_path, dpi=300)  # Save with high resolution
plt.show()

print(f"ROC Curve plot saved at {roc_curve_plot_path}")

# Evaluate the model precision-recall , average precision and mAP
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test_encoded, axis=1)

# Calculate precision-recall and average precision for each class

# Number of classes
num_classes = y_test_encoded.shape[1]

# Store AP values for each class
average_precisions = {}

# Plot Precision-Recall curves
plt.figure(figsize=(10, 7))
for i in range(num_classes):
    precision, recall, _ = precision_recall_curve(y_test_encoded[:, i], y_pred[:, i])
    ap = average_precision_score(y_test_encoded[:, i], y_pred[:, i])
    average_precisions[i] = ap

    # Plot PR curve for each class
    plt.plot(recall, precision, label=f"Class {label_encoder.classes_[i]} (AP={ap:.2f})")

# Calculate mAP
mAP = np.mean(list(average_precisions.values()))
print(f"Mean Average Precision (mAP): {mAP:.4f}")

# Finalize PR plot
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()

# Save the PR curve plot
pr_curve_path = os.path.join(WORKING_DIR, f'pr_curve_{timestamp}.png')
plt.savefig(pr_curve_path)
plt.show()
print(f"Precision-Recall curve saved at {pr_curve_path}")