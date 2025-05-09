import dlib
import torch.nn as nn

import logging

from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy
from sklearn.metrics import roc_auc_score,roc_curve
import random

import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
from albumentations import (
    Compose, ImageCompression, GaussNoise, GaussianBlur,
    RandomBrightnessContrast, CoarseDropout, HueSaturationValue,
    RGBShift, Resize, CenterCrop, HorizontalFlip, Rotate, Normalize
)
from albumentations.pytorch import ToTensorV2

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import pywt
import os
import cv2
import torch.multiprocessing as mp

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



class SimpleDiffusionModel(nn.Module):
    def __init__(self, timesteps=1000):
        super(SimpleDiffusionModel, self).__init__()
        self.timesteps = timesteps

    def forward_diffusion(self, x, t):
        noise = torch.randn_like(x)  # Gaussian noise
        alpha_t = torch.exp(-0.5 * t.float() / self.timesteps)
        noisy_x = alpha_t * x + (1 - alpha_t) * noise
        return noisy_x, noise
class WaveletDecomposition(nn.Module):
    def __init__(self, wavelet='haar', diffusion_model=None):
        super(WaveletDecomposition, self).__init__()
        self.wavelet = wavelet
        self.diffusion_model = diffusion_model

    def apply_diffusion(self, components, t=None):
        if self.diffusion_model:
            if t is None:
                # Choose a small t to ensure subtle noise
                t = torch.randint(0, int(self.diffusion_model.timesteps * 0.2), (1,))
            return self.diffusion_model.forward_diffusion(components, t)[0]
        return components

    def localized_perturbation(self, components, scale_factor=0.95):
        h, w = components.shape[-2:]
        region_h = h // 2
        region_w = w // 2
        x = random.randint(0, h - region_h)
        y = random.randint(0, w - region_w)

        perturbed_matrix = components.clone()
        perturbed_matrix[..., x:x + region_h, y:y + region_w] *= scale_factor
        return perturbed_matrix

    def random_noise_injection(self, components, noise_level=0.002):
        noise = torch.randn_like(components) * noise_level
        return components + noise

    def coefficient_thresholding(self, components, threshold=0.01):
        thresholded_matrix = torch.where(torch.abs(components) < threshold, torch.tensor(0.0, device=components.device), components)
        return thresholded_matrix

    def apply_augmentations(self, components):
        if not self.training:
            return components  # No augmentations during evaluation/inference

        # 70% chance to keep the original components
        if random.random() > 0.7:
            return components  # Return original

        # List of augmentations
        augmentations = [
            self.localized_perturbation,
            self.random_noise_injection,
            self.coefficient_thresholding,
            self.apply_diffusion  # Apply diffusion as an augmentation
        ]

        # Choose one or two augmentations to apply
        if random.random() > 0.7:
            augmentation = random.choice(augmentations)
            augmented_components = augmentation(components)
        else:
            augmentation1, augmentation2 = random.sample(augmentations, 2)
            augmented_components = augmentation1(components)
            augmented_components = augmentation2(augmented_components)

        return augmented_components

    def forward(self, x):
        coeffs = pywt.wavedec2(x.cpu().numpy(), self.wavelet, level=4)
        coeffs = [torch.tensor(c).to(x.device) for c in coeffs]
        # Level 4 (the coarsest level)
        input_l4_LL = coeffs[0]  # Approximation coefficients at level 4, unsqueeze to match dimensions
        input_l4 = torch.cat(
            [ coeffs[1][0].unsqueeze(1), coeffs[1][1].unsqueeze(1), coeffs[1][2].unsqueeze(1)], dim=1)
        input_l4 = self.apply_augmentations(input_l4)
        input_l4_LL = self.apply_augmentations(input_l4_LL)
        input_l4 = (input_l4_LL,input_l4.reshape(input_l4.shape[0],-1,input_l4.shape[3],input_l4.shape[4]))

        # Level 3
        input_l3_LL = coeffs[1][0] # Approximation coefficients (LL) at level 3
        input_l3 = torch.cat(
            [ coeffs[2][0].unsqueeze(1), coeffs[2][1].unsqueeze(1), coeffs[2][2].unsqueeze(1)], dim=1)
        input_l3 = self.apply_augmentations(input_l3)
        input_l3_LL = self.apply_augmentations(input_l3_LL)
        input_l3 = (input_l3_LL, input_l3.reshape(input_l3.shape[0],-1,input_l3.shape[3],input_l3.shape[4]))

        # Level 2
        input_l2_LL = coeffs[2][0]  # Approximation coefficients (LL) at level 2
        input_l2 = torch.cat(
            [ coeffs[3][0].unsqueeze(1), coeffs[3][1].unsqueeze(1), coeffs[3][2].unsqueeze(1)], dim=1)
        input_l2 = self.apply_augmentations(input_l2)
        input_l2_LL = self.apply_augmentations(input_l2_LL)
        input_l2 = (input_l2_LL, input_l2.reshape(input_l2.shape[0],-1,input_l2.shape[3],input_l2.shape[4]))



        # Level 1 (the finest level)
        input_l1_LL = coeffs[3][0] # Approximation coefficients (LL) at level 1
        input_l1 = torch.cat(
            [ coeffs[4][0].unsqueeze(1), coeffs[4][1].unsqueeze(1), coeffs[4][2].unsqueeze(1)], dim=1)
        input_l1 = self.apply_augmentations(input_l1)
        input_l1_LL = self.apply_augmentations(input_l1_LL)

        input_l1 = (input_l1_LL, input_l1.reshape(input_l1.shape[0],-1,input_l1.shape[3],input_l1.shape[4]))




        # Return the four levels for further processing in the network
        return input_l1, input_l2, input_l3, input_l4


class CDCBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, theta=0.2):
        super(CDCBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              bias=False)
        self.theta = nn.Parameter(torch.tensor(theta, dtype=torch.float32))  # trainable parameter theta

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()


    def forward(self, x):
        # Standard convolution
        normal_conv = self.conv(x)

        # Central difference convolution
        kernel_diff = self.conv.weight.sum(dim=(2, 3), keepdim=True)
        central_diff = F.conv2d(x, kernel_diff, bias=None, stride=self.conv.stride, padding=0, groups=self.conv.groups)

        # Combine standard convolution with central difference
        return self.relu(self.bn(normal_conv - self.theta * central_diff))


class CDCBlockDownsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, theta=0.2):
        super(CDCBlockDownsample, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              bias=False)
        self.theta = nn.Parameter(torch.tensor(theta, dtype=torch.float32))  # trainable parameter theta

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Standard convolution
        normal_conv = self.conv(x)

        # Central difference convolution
        kernel_diff = self.conv.weight.sum(dim=(2, 3), keepdim=True)
        central_diff = F.conv2d(x, kernel_diff, bias=None, stride=self.conv.stride, padding=0, groups=self.conv.groups)

        # Combine standard convolution with central difference
        return self.pool(self.relu(self.bn(normal_conv - self.theta * central_diff)))


## The upper CDC block can be changed with the below CNN if required for experiments.
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class ConvBlockDownSample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlockDownSample, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return  self.pool (self.relu(self.bn(self.conv(x))))





class CustomImageDataset(Dataset):
    def __init__(self, csv_file, transform=None,predictor=None):
        self.dataframe = pd.read_csv(csv_file)
        self.transform = transform
        self.predictor = predictor



    def __len__(self):
        return len(self.dataframe)




    def get_facial_landmarks(self, image):
        """Assume image contains a single face, detect landmarks directly."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        rect = dlib.rectangle(0, 0, image.shape[1], image.shape[0])  # Use the entire image as the face region
        shape = self.predictor(gray, rect)
        landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
        return landmarks
    def apply_perturbations_to_landmarks(self, image, landmarks, region_size=10):
        for (x, y) in landmarks:
            # Define the region around the landmark
            x_start, y_start = max(0, x - region_size // 2), max(0, y - region_size // 2)
            x_end, y_end = min(image.shape[1], x + region_size // 2), min(image.shape[0], y + region_size // 2)

            # Extract the region of interest (ROI)
            region = image[y_start:y_end, x_start:x_end]

            # Randomly choose between Gaussian and Speckle noise
            # noise_type = random.choice(['gaussian', 'speckle'])
            noise_type = "gaussian"
            if noise_type == 'gaussian':
                # Apply subtle Gaussian noise
                noise = np.random.normal(0, 0.02, region.shape)
                region = region + noise
            # elif noise_type == 'speckle':
            #     # Apply subtle Speckle noise (multiplicative noise)
            #     noise = np.random.normal(0, 0.02, region.shape)
            #     region = region + region * noise

            # Make the image writable by creating a copy
            image = image.copy()
            # Place the modified region back into the image
            image[y_start:y_end, x_start:x_end] = region

        return image




    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx, 0]
        img_path = os.path.normpath(img_name)
        image = Image.open(img_path).convert('RGB')  # Ensure 3 channels
        label = torch.tensor(self.dataframe.iloc[idx, 1], dtype=torch.float32)

        # Convert PIL image to numpy array
        image = np.asarray(image)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        image_wavelet_transform = image
        return image_wavelet_transform, label

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(x)
        avg_out = self.fc1(avg_out)
        avg_out = self.relu(avg_out)
        avg_out = self.fc2(avg_out)
        return x * self.sigmoid(avg_out)
class WaveletCNN(nn.Module):
    def __init__(self):
        super(WaveletCNN, self).__init__()
        self.wavelet_decompose = WaveletDecomposition()

        # Define convolutional blocks for each level
        self.level1_conv1_ll = CDCBlock(3, 64)
        self.level1_conv1_h = CDCBlockDownsample(9, 64)
        self.level1_conv1 = CDCBlock(128, 128)
        self.level1_conv2 = CDCBlock(128, 128, stride=2)
        self.level1_ca = ChannelAttention(128)
        self.adaptive_pool_level1 = nn.AdaptiveAvgPool2d((14, 14))



        self.level2_conv1_ll = CDCBlock(3, 128)
        self.level2_conv1_h = CDCBlockDownsample(9, 128)
        self.level2_conv1 = CDCBlock(256, 128)
        self.level2_conv2 = CDCBlock(256, 256, stride=2)
        self.level2_ca = ChannelAttention(256)
        self.adaptive_pool_level2 = nn.AdaptiveAvgPool2d((14, 14))



        self.level3_conv1_ll = CDCBlock(3, 128)
        self.level3_conv1_h = CDCBlockDownsample(9, 128)
        self.level3_conv1 = CDCBlock(256, 256)
        self.level3_conv2 = CDCBlock(512, 256)
        self.level3_ca = ChannelAttention(256)
        self.adaptive_pool_level3 = nn.AdaptiveAvgPool2d((14, 14))


        self.level4_conv1_ll = CDCBlock(3, 128)
        self.level4_conv1_h = CDCBlock(9, 128)
        self.level4_conv1 = CDCBlock(256, 256)
        self.level4_conv2 = CDCBlock(512, 512)
        self.level4_conv3 = CDCBlock(512, 1024)
        self.level4_ca = ChannelAttention(1024)


        #final concat


        # Final convolutional block
        self.final_conv = ConvBlock(1664, 512)

        # Fully connected layers
        self.fc1 = nn.Linear(512 * 14 * 14, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, 2)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x, return_features = False):
        # Perform wavelet decomposition
        decomp = self.wavelet_decompose(x)

        # Process each level separately and combine them progressively
        x1_ll = self.level1_conv1_ll(decomp[0][0])
        x1_h = self.level1_conv1_h(decomp[0][1])
        x1 = torch.cat((x1_ll, x1_h), dim=1)
        x1 = self.level1_conv1(x1)
        x1 = self.level1_conv2(x1)
        x1 = self.level1_ca(x1)  # Apply channel attention
        x1_downsampled = self.adaptive_pool_level1(x1)




        x2_ll = self.level2_conv1_ll(decomp[1][0])
        x2_h = self.level2_conv1_h(decomp[1][1])
        x2 = torch.cat((x2_ll, x2_h), dim=1)
        x2 = self.level2_conv1(x2)
        x2 = torch.cat((x1, x2), dim=1)  # Concatenate level 1 output with level 2 input
        x2 = self.level2_conv2(x2)
        x2 = self.level2_ca(x2)  # Apply channel attention
        x2_downsampled = self.adaptive_pool_level1(x2)




        x3_ll = self.level3_conv1_ll(decomp[2][0])
        x3_h = self.level3_conv1_h(decomp[2][1])
        x3 = torch.cat((x3_ll, x3_h), dim=1)
        x3 = self.level3_conv1(x3)
        x3 = torch.cat((x2, x3), dim=1)  # Concatenate level 2 output with level 3 input
        x3 = self.level3_conv2(x3)
        x3 = self.level3_ca(x3)  # Apply channel attention
        x3_downsampled = self.adaptive_pool_level1(x3)



        x4_ll = self.level4_conv1_ll(decomp[3][0])
        x4_h = self.level4_conv1_h(decomp[3][1])
        x4 = torch.cat((x4_ll, x4_h), dim=1)
        x4 = self.level4_conv1(x4)
        x4 = torch.cat((x3, x4), dim=1)  # Concatenate level 3 output with level 4 input
        x4 = self.level4_conv2(x4)
        x4 = self.level4_conv3(x4)
        x4 = self.level4_ca(x4)  # Apply channel attention

        x_concat = torch.cat((x1_downsampled, x2_downsampled,x3_downsampled,x4), dim=1)

        # Final convolution
        x4 = self.final_conv(x_concat)

        # Flatten and pass through fully connected layers
        x4 = x4.view(x4.size(0), -1)
        x4 = F.relu(self.fc1(x4))
        x4 = self.dropout(x4)
        x4 = F.relu(self.fc2(x4))
        if return_features:
            return x4
        x4 = self.dropout(x4)
        x4 = self.fc3(x4)

        return x4

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



probability = 0.4
transform = Compose([
    Resize(224, 224),
    HorizontalFlip(p=probability),
    Rotate(limit=60, p=probability),
    ImageCompression(quality_lower=10, quality_upper=70, p=0.7),
    GaussNoise(var_limit=(10.0, 50.0), p=probability),
    GaussianBlur(blur_limit=1, p=probability),
    RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=probability),
    CoarseDropout(max_holes=4, max_height=4, max_width=4, fill_value=0, p=probability),
    HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10, p=0.4),
    RGBShift(r_shift_limit=5, g_shift_limit=5, b_shift_limit=5, p=0.4),
    Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

probability = 0.2
transform_val = Compose([
    Resize(224, 224),
    Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])


# Initialize logging
logging.basicConfig(filename='waveletCNN.log', level=logging.INFO,force= True)

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



def train(rank, world_size, num_epochs, train_csv_file, val_csv_file,batch_size, patience, lr, weight_decay, model_path):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    batch_size = batch_size


    # Load the face detector and facial landmark predictor
    predictor = dlib.shape_predictor('dlib_landmarks/shape_predictor_68_face_landmarks.dat')

    # Load datasets
    train_dataset = CustomImageDataset(csv_file=train_csv_file, transform=transform,predictor = predictor)
    val_dataset = CustomImageDataset(csv_file=val_csv_file, transform=transform_val, predictor = predictor)

    # Create dataloaders
    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, drop_last=True)

    # Initialize the model
    model = WaveletCNN()

    # Define loss functions and optimizer
    criterion = SoftTargetCrossEntropy().to(rank)
    # criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
    focal_loss = FocalLoss(alpha=0.25, gamma=2).to(rank)
    optimizer = optim.AdamW(model.parameters(), lr= lr, weight_decay=weight_decay)

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

    def one_hot_encode(labels, num_classes):
        device = labels.device
        return torch.eye(num_classes, device=device)[labels]

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

            outputs = model(images)
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
                outputs = model(images)

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

        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
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
        plt.savefig('training_validation_loss_plot_wavelet_cnn.png')
        plt.show()

    destroy_process_group()


# Evaluate on test dataset
def evaluate_model(rank, world_size, test_csv_file, batch_size, model_path):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    batch_size = batch_size

    val_test_transform = Compose([
        Resize(224, 224),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    test_dataset = CustomImageDataset(csv_file=test_csv_file, transform=val_test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             sampler=DistributedSampler(test_dataset))

    model = WaveletCNN()
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
            outputs = model(images)
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
