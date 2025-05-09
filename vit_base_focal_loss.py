import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import ViTFeatureExtractor, ViTForImageClassification
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy
from sklearn.metrics import roc_auc_score,roc_curve
from PIL import Image
from albumentations import (
    Compose, ImageCompression, GaussNoise, GaussianBlur,
    RandomBrightnessContrast, CoarseDropout, HueSaturationValue,
    RGBShift, Resize, CenterCrop, HorizontalFlip, Rotate, Normalize
)





import argparse

def parse_args():
    parser = argparse.ArgumentParser()


    parser.add_argument('--train', action='store_true', help='Run training')
    parser.add_argument('--eval', action='store_true', help='Run evaluation')

    parser.add_argument('--world_size', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--train_csv', type=str)
    parser.add_argument('--val_csv', type=str)
    parser.add_argument('--test_csv', type=str)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--model_path', type=str, default='model.pth')
    parser.add_argument('--patience', type=int, default=5)

    # Parse arguments
    args = parser.parse_args()

    # Validate only *after* parsing
    if args.train:
        if not args.train_csv or not args.val_csv or not args.model_path:
            parser.error('--train requires --train_csv, --val_csv, and --model_path')

    if args.eval:
        if not args.test_csv or not args.model_path:
            parser.error('--eval requires --test_csv and --model_path')

    return args


# Custom Dataset
class CustomImageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.dataframe = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx, 0]
        img_path = os.path.normpath(img_name)
        image = Image.open(img_path).convert('RGB')
        label = torch.tensor(self.dataframe.iloc[idx, 1], dtype=torch.float32)

        image = np.asarray(image)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, label

# Focal Loss Class
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

# Custom Model
class ViTForDeepfakeDetection(nn.Module):
    def __init__(self, vit_model_name='google/vit-base-patch16-224', num_classes=2):
        super(ViTForDeepfakeDetection, self).__init__()

        # Load the ViTForImageClassification model
        self.vit_model = ViTForImageClassification.from_pretrained(vit_model_name)

        # Replace the classifier layer to match the number of output classes
        self.vit_model.classifier = nn.Linear(self.vit_model.config.hidden_size, num_classes)

        # Freeze all layers except the specified parts
        for name, param in self.vit_model.named_parameters():
            if not any(x in name for x in ['intermediate', 'output', 'layernorm', 'classifier']):
                param.requires_grad = False

    def forward(self, x):
        # Extract features using the ViT model
        vit_output = self.vit_model.vit(x).last_hidden_state[:, 0, :]  # [CLS] token
        logits = self.vit_model.classifier(vit_output)
        return logits, vit_output

# Define transformations using Albumentations
probability = 0.2
transform = Compose([
    Resize(224, 224),
    CenterCrop(224, 224),
    HorizontalFlip(p=probability),
    Rotate(limit=15, p=probability),
    ImageCompression(quality_lower=30, quality_upper=100, p=probability),
    GaussNoise(var_limit=(10.0, 50.0), p=probability),
    GaussianBlur(blur_limit=3, p=probability),
    RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=probability),
    CoarseDropout(max_holes=8, max_height=8, max_width=8, fill_value=0, p=probability),
    HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=probability),
    RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=probability),
    Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# Initialize logging
logging.basicConfig(filename='vit_training_focal_loss.log', level=logging.INFO)

# Initialize Mixup/CutMix
mixup_fn = Mixup(
    mixup_alpha=1.0,
    cutmix_alpha=1.0,
    cutmix_minmax=None,
    prob=0.5,
    switch_prob=0.5,
    mode='batch',
    label_smoothing=0.1,
    num_classes=2
)

def one_hot_encode(labels, num_classes):
    device = labels.device
    return torch.eye(num_classes, device=device)[labels]

# Training function with Mixup/CutMix from timm
def train(rank, world_size, num_epochs, train_csv_file, val_csv_file, batch_size, patience, lr, weight_decay, model_path):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    batch_size = batch_size

    # Load datasets
    train_dataset = CustomImageDataset(csv_file=train_csv_file, transform=transform)
    val_dataset = CustomImageDataset(csv_file=val_csv_file, transform=transform)

    # Create dataloaders
    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, drop_last=True)

    # Initialize the model
    model = ViTForDeepfakeDetection(vit_model_name='google/vit-base-patch16-224', num_classes=2)

    # Define loss functions and optimizer
    criterion = SoftTargetCrossEntropy().to(rank)
    focal_loss = FocalLoss(alpha=0.35, gamma=4).to(rank)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    model = DDP(model.to(rank), device_ids=[rank])

    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Track training and validation loss
    train_loss_history = []
    train_total_loss_history = []
    val_loss_history = []
    val_total_loss_history = []
    best_val_loss = float('inf')
    patience_counter = 0
    patience = patience

    print("Training Started")
    logging.info("Training Started")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_focal_loss = 0.0
        running_total_loss = 0.0
        train_correct = 0
        total_train = 0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(rank), labels.to(rank)
            labels = labels.long()

            optimizer.zero_grad()

            if mixup_fn:
                images, labels = mixup_fn(images, labels)

            outputs, features = model(images)
            loss = criterion(outputs, labels)
            loss_focal = focal_loss(outputs, labels)

            # Combine the classification loss and focal loss
            loss_total = 0.8 * loss + 0.2 * loss_focal

            loss_total.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_focal_loss += loss_focal.item() * images.size(0)
            running_total_loss += loss_total.item() * images.size(0)

            # Calculate training accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)

            # Convert one-hot encoded labels to class indices
            labels = torch.argmax(labels, dim=1)

            train_correct += (predicted == labels).sum().item()

            log_message = (f"Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(train_loader)}], "
                           f"Train Loss: {loss.item():.4f}, Focal Loss: {loss_focal.item():.4f}, "
                           f"Total Loss: {loss_total.item():.4f}")
            print(log_message)
            logging.info(log_message)

        scheduler.step()
        average_train_loss = running_loss / (len(train_loader) * batch_size)
        average_focal_loss = running_focal_loss / (len(train_loader) * batch_size)
        average_total_loss = running_total_loss / (len(train_loader) * batch_size)
        train_loss_history.append(average_train_loss)
        train_total_loss_history.append(average_total_loss)
        train_accuracy = 100.0 * train_correct / total_train

        # Validation loop
        model.eval()
        running_val_loss = 0.0
        running_val_focal_loss = 0.0
        running_val_total_loss = 0.0
        val_correct = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(rank), labels.to(rank)
                labels = labels.long()
                outputs, features = model(images)

                # Convert labels to one-hot encoding for loss calculation
                labels_one_hot = one_hot_encode(labels, num_classes=2).to(rank)

                val_loss = criterion(outputs, labels_one_hot)
                running_val_loss += val_loss.item() * images.size(0)

                val_focal_loss = focal_loss(outputs, labels)
                running_val_focal_loss += val_focal_loss.item() * images.size(0)

                # Combine the classification loss and focal loss
                val_total_loss = 0.8 * val_loss + 0.2 * val_focal_loss
                running_val_total_loss += val_total_loss.item() * images.size(0)

                # Calculate validation accuracy
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)

                val_correct += (predicted == labels).sum().item()

        average_val_loss = running_val_loss / (len(val_loader) * batch_size)
        average_val_focal_loss = running_val_focal_loss / (len(val_loader) * batch_size)
        average_val_total_loss = running_val_total_loss / (len(val_loader) * batch_size)
        val_loss_history.append(average_val_loss)
        val_total_loss_history.append(average_val_total_loss)
        val_accuracy = 100.0 * val_correct / total_val

        log_message = (f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {average_train_loss:.4f}, "
                       f"Focal Loss: {average_focal_loss:.4f}, Total Loss: {average_total_loss:.4f}, "
                       f"Training Accuracy: {train_accuracy:.2f}%, Validation Loss: {average_val_loss:.4f}, "
                       f"Validation Focal Loss: {average_val_focal_loss:.4f}, "
                       f"Validation Total Loss: {average_val_total_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
        print(log_message)
        logging.info(log_message)

        if average_val_total_loss < best_val_loss:
            best_val_loss = average_val_total_loss
            patience_counter = 0
            if rank == 0:
                torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1

        if patience_counter > patience:
            log_message = f"Early stopping after {patience} epochs with no improvement."
            print(log_message)
            logging.info(log_message)
            break

    if rank == 0:
        plt.plot(train_loss_history, label='Training Loss')
        plt.plot(train_total_loss_history, label='Training Total Loss')
        plt.plot(val_total_loss_history, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.savefig('training_validation_loss_plot_focal.png')
        plt.show()

    destroy_process_group()

# Evaluation function
def evaluate_model(rank, world_size, test_csv_file,batch_size, model_path):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    batch_size = batch_size

    val_test_transform = A.Compose([
        A.Resize(224, 224),
        A.CenterCrop(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    test_dataset = CustomImageDataset(csv_file=test_csv_file, transform=val_test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             sampler=DistributedSampler(test_dataset))

    model = ViTForDeepfakeDetection(vit_model_name='google/vit-base-patch16-224')
    model = DDP(model.to(rank), device_ids=[rank])

    state_dict = torch.load(model_path, map_location={'cuda:0': f'cuda:{rank}'})
    model.load_state_dict(state_dict)

    model.eval()
    total_correct = 0
    total_samples = 0

    predicted_probs = []
    true_labels = []

    print("start evaluating")
    logging.info("Start evaluating")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(rank), labels.to(rank)
            labels = labels.long()
            outputs, features = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)

            total_correct += (predicted == labels).sum().item()

            probs = torch.nn.functional.softmax(outputs, dim=1)[:, 1]
            predicted_probs.extend(probs.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    auc = roc_auc_score(true_labels, predicted_probs)
    accuracy = 100 * total_correct / total_samples


    # Calculate EER
    fpr, tpr, thresholds = roc_curve(true_labels, predicted_probs)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.abs(fnr - fpr))]
    eer = fpr[np.nanargmin(np.abs(fnr - fpr))]


    log_message = f"AUC: {auc:.4f}, Test Accuracy: {accuracy:.2f}%, EER: {eer:.4f} at threshold: {eer_threshold:.4f}"
    print(log_message)

    # log_message = f"AUC: {auc:.4f}, Test Accuracy: {accuracy:.2f}%"
    # print(log_message)
    logging.info(log_message)

    destroy_process_group()

if __name__ == '__main__':
    args = parse_args()
    # TRAINING
    mp.spawn(
        train,
        args=(
            args.world_size,
            args.num_epochs,
            args.train_csv,
            args.val_csv,
            args.batch_size,
            args.patience,
            args.lr,
            args.weight_decay,
            args.model_path
        ),
        nprocs=args.world_size
    )

    # EVALUATION
    mp.spawn(
        evaluate_model,
        args=(
            args.world_size,
            args.test_csv,
            args.batch_size,
            args.model_path
        ),
        nprocs=args.world_size
    )
