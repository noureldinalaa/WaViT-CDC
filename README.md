# WaViT-CDC: Wavelet Vision Transformer with Central Difference Convolutions for Spatial-Frequency Deepfake Detection
![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)

**WaViT-CDC** is a novel deepfake detection framework that fuses frequency and spatial domain information using a Wavelet-Central Difference Convolution module and a Vision Transformer, connected via a Frequency-Spatial Feature Fusion Attention mechanism. It achieves competitive  cross-dataset generalization performance on Celeb-DF and WildDeepfake.

---

## Architecture Overview
![WaViT-CDC Architecture](https://github.com/noureldinalaa/WaViT-CDC/blob/main/WaviT-CDC_Architecture)


---

## Datasets Used
- **FaceForensics++ (LQ & HQ)**
- **Celeb-DF (V2)**
- **WildDeepfake**


CSV files that list image paths and labels must be prepared. Three CSV files (Training, Validation, Testing) were created following the dataset protocols for dataset splitting. Sample CSV structure:

```csv
path,label
/path/to/image1.jpg,0
/path/to/image2.jpg,1
```
---

## Installation

### 1. Clone the repository

```
git clone https://github.com/noureldinalaa/WaViT-CDC.git
cd Wavit-CDC
```

### 2. Install Python dependencies
Experiments were run on **Ubuntu 22.04.5** with Python **3.10**. 
Anaconda is recommended to manage the Python environment.
```
conda create -n wavit-cdc python=3.10 -y
conda activate wavit-cdc

```
Install the dependencies
```
pip install -r requirements.txt
```

To install **dlib** :
```
pip install cmake
pip install dlib -vvv
```

---
## Training
The model was trained using **4× NVIDIA RTX A6000 GPUs** with **PyTorch Distributed Data Parallel (DDP)**.

```
python wavit_cdc.py \
  --train \
  --world_size 4 \
  --num_epochs 10 \
  --batch_size 128 \
  --train_csv path_to_training_data.csv \
  --val_csv path_to_validation_data.csv \
  --lr 1e-4 \
  --weight_decay 0.02 \
  --patience 5 \
  --model_path path_to_save_model.pth

```

---

## Evaluation
```
python wavit_cdc.py \
  --eval \
  --world_size 4 \
  --batch_size 64 \
  --test_csv path_to_testing_data.csv \
  --model_path saved_trained_model_path.pth

```
---

## WaViT-CDC Results (Cross-Dataset Evaluation)

| Training Dataset | Testing Dataset       | AUC (%) | EER ↓   |
|------------------|---------------|---------|---------|
| FF++ (LQ)        | Celeb-DF      | 78.60   | 0.282   |
|                  | WildDeepfake  | 75.91   | 0.307   |
| FF++ (HQ)        | Celeb-DF      | 81.62   | 0.258   |
|                  | WildDeepfake  | 81.23   | 0.266   |
---
#### Facial Landmark Model

This work uses **Dlib’s 68-point facial landmark detector**.

The pre-trained model file can be downloaded from the official Dlib website:

[shape_predictor_68_face_landmarks.dat.bz2](https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)






