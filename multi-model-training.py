import os
import pandas as pd
import numpy as np
from datetime import datetime
import torch
import torchvision
import tensorflow as tf
from torchvision.models import vit_b_32, ViT_B_32_Weights
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from ultralytics import YOLO
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import cv2

# Paths
TRAIN_PATH = '/kaggle/input/coco-2017-dataset/coco2017/train2017'
VAL_PATH = '/kaggle/input/coco-2017-dataset/coco2017/val2017'
ANNOTATIONS_PATH = '/kaggle/input/coco-2017-dataset/coco2017/annotations'
FILTERED_DATASET = '/kaggle/input/filtered-coco-dataset'
WORKING_DIR = '/kaggle/working'
FILTERED_CATEGORIES = ['person', 'cat', 'dog']

class DataProcessor:
    def __init__(self, train_csv_path, test_csv_path):
        self.train_data = pd.read_csv(train_csv_path)
        self.test_data = pd.read_csv(test_csv_path)
        self.label_encoder = LabelEncoder()
        
    def preprocess_image(self, image_path, target_size=(224, 224)):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, target_size)
        image = image / 255.0
        return image

    def load_data(self, data, path_prefix):
        images = []
        labels = []
        for _, row in data.iterrows():
            image_path = os.path.join(path_prefix, row['image'])
            images.append(self.preprocess_image(image_path))
            labels.append(row['category_id'])
        
        images = np.array(images)
        labels = np.array(labels)
        
        # Encode labels
        labels_encoded = self.label_encoder.fit_transform(labels)
        labels_one_hot = to_categorical(labels_encoded)
        
        return images, labels_one_hot

class ModelTrainer:
    def __init__(self, data_processor):
        self.data_processor = data_processor
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Load and process data
        self.X_train, self.y_train = self.data_processor.load_data(
            self.data_processor.train_data, TRAIN_PATH)
        self.X_test, self.y_test = self.data_processor.load_data(
            self.data_processor.test_data, TRAIN_PATH)
        
    def train_mobilenet(self):
        print("Training MobileNetV2...")
        
        # Build the model
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        
        # Unfreeze some layers for fine-tuning
        for layer in base_model.layers[-30:]:
            layer.trainable = True
            
        # Add classification head
        inputs = Input(shape=(224, 224, 3))
        x = base_model(inputs)
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
        x = Dropout(0.3)(x)
        outputs = Dense(3, activation='softmax')(x)
        model = Model(inputs, outputs)
        
        # Compile model
        optimizer = Adam(learning_rate=0.0001)
        model.compile(optimizer=optimizer,
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        
        # Train model
        history = model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_test, self.y_test),
            epochs=30,
            batch_size=32,
            verbose=1
        )
        
        # Save model
        model_path = os.path.join(WORKING_DIR, f'mobilenet_v2_coco_{self.timestamp}.keras')
        model.save(model_path)
        
        return history.history, model
        
    def train_faster_rcnn(self):
        print("Training Faster R-CNN...")
        
        # Initialize Faster R-CNN
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        model.eval()
        
        # Convert data to PyTorch tensors
        X_train_torch = torch.FloatTensor(self.X_train)
        y_train_torch = torch.FloatTensor(self.y_train)
        
        # Train model
        history = self._train_torch_model(model, X_train_torch, y_train_torch, "faster_rcnn")
        return history, model
    
    def train_yolo(self):
        print("Training YOLOv8...")
        
        # Initialize YOLOv8
        model = YOLO('yolov8n.pt')
        
        # Train model
        history = model.train(
            data='coco.yaml',
            epochs=30,
            imgsz=224,
            batch=32
        )
        return vars(history), model
    
    def train_vit(self):
        print("Training ViT...")
        
        # Initialize ViT
        model = vit_b_32(weights=ViT_B_32_Weights.DEFAULT)
        model.eval()
        
        # Convert data to PyTorch tensors
        X_train_torch = torch.FloatTensor(self.X_train)
        y_train_torch = torch.FloatTensor(self.y_train)
        
        # Train model
        history = self._train_torch_model(model, X_train_torch, y_train_torch, "vit")
        return history, model

    def _train_torch_model(self, model, X_train, y_train, model_name):
        # Common PyTorch training loop implementation
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        criterion = torch.nn.CrossEntropyLoss()
        
        for epoch in range(30):
            # Training
            model.train()
            train_loss = 0
            train_correct = 0
            
            for i in range(0, len(X_train), 32):
                batch_x = X_train[i:i+32]
                batch_y = y_train[i:i+32]
                
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                _, true_classes = torch.max(batch_y, 1)
                train_correct += (predicted == true_classes).sum().item()
            
            # Calculate metrics
            train_loss = train_loss / len(X_train)
            train_acc = train_correct / len(X_train)
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            
            print(f'Epoch {epoch+1}/{30} - loss: {train_loss:.4f} - acc: {train_acc:.4f}')
        
        return history

class MetricsEvaluator:
    def __init__(self, working_dir):
        self.working_dir = working_dir
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
    def plot_metrics(self, histories, models, X_test, y_test):
        self._plot_accuracy_loss(histories)
        self._plot_roc_curves(models, X_test, y_test)
        self._plot_precision_recall(models, X_test, y_test)
        
    def _plot_accuracy_loss(self, histories):
        plt.figure(figsize=(15, 5))
        
        # Accuracy plot
        plt.subplot(1, 2, 1)
        for model_name, history in histories.items():
            if 'accuracy' in history:  # MobileNetV2 format
                plt.plot(history['accuracy'], label=f'{model_name} Train')
                plt.plot(history['val_accuracy'], label=f'{model_name} Val')
            else:  # PyTorch format
                plt.plot(history['train_acc'], label=f'{model_name} Train')
                plt.plot(history['val_acc'], label=f'{model_name} Val')
        plt.title('Model Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Loss plot
        plt.subplot(1, 2, 2)
        for model_name, history in histories.items():
            if 'loss' in history:  # MobileNetV2 format
                plt.plot(history['loss'], label=f'{model_name} Train')
                plt.plot(history['val_loss'], label=f'{model_name} Val')
            else:  # PyTorch format
                plt.plot(history['train_loss'], label=f'{model_name} Train')
                plt.plot(history['val_loss'], label=f'{model_name} Val')
        plt.title('Model Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.savefig(os.path.join(self.working_dir, f'combined_metrics_{self.timestamp}.png'))
        plt.close()
        
    def _plot_roc_curves(self, models, X_test, y_test):
        plt.figure(figsize=(10, 8))
        
        for model_name, model in models.items():
            if isinstance(model, tf.keras.Model):  # MobileNetV2
                y_pred = model.predict(X_test)
            else:  # PyTorch models
                model.eval()
                with torch.no_grad():
                    y_pred = model(torch.FloatTensor(X_test)).numpy()
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_test.ravel(), y_pred.ravel())
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
            
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend(loc='lower right')
        
        plt.savefig(os.path.join(self.working_dir, f'combined_roc_{self.timestamp}.png'))
        plt.close()
        
    def _plot_precision_recall(self, models, X_test, y_test):
        plt.figure(figsize=(10, 8))
        
        for model_name, model in models.items():
            if isinstance(model, tf.keras.Model):  # MobileNetV2
                y_pred = model.predict(X_test)
            else:  # PyTorch models
                model.eval()
                with torch.no_grad():
                    y_pred = model(torch.FloatTensor(X_test)).numpy()
            
            # Calculate metrics for each class
            n_classes = y_test.shape[1]
            for i in range(n_classes):
                precision, recall, _ = precision_recall_curve(y_test[:, i], y_pred[:, i])
                ap = average_precision_score(y_test[:, i], y_pred[:, i])
                plt.plot(recall, precision, 
                        label=f'{model_name} - Class {i} (AP = {ap:.2f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves Comparison')
        plt.legend(loc='lower left')
        
        plt.savefig(os.path.join(self.working_dir, f'combined_pr_{self.timestamp}.png'))
        plt.close()
        
def main():
    # Initialize data processor
    data_processor = DataProcessor(
        os.path.join(FILTERED_DATASET, 'train_data.csv'),
        os.path.join(FILTERED_DATASET, 'test_data.csv')
    )
    
    # Initialize model trainer
    trainer = ModelTrainer(data_processor)
    
    # Train models
    models = {}
    histories = {}
    
    # Train MobileNetV2
    histories['mobilenet'], models['mobilenet'] = trainer.train_mobilenet()
    
    # Train Faster R-CNN
    histories['faster_rcnn'], models['faster_rcnn'] = trainer.train_faster_rcnn()
    
    # Train YOLOv8
    histories['yolo'], models['yolo'] = trainer.train_yolo()
    
    # Train ViT
    histories['vit'], models['vit'] = trainer.train_vit()
    
    # Evaluate metrics
    evaluator = MetricsEvaluator(WORKING_DIR)
    evaluator.plot_metrics(histories, models, trainer.X_test, trainer.y_test)
    
    print("Training and evaluation complete. Results saved in working directory.")

if __name__ == "__main__":
    main()        