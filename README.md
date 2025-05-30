# Semi-Supervised and Temporally-Aware Segmentation of High-Resolution Transmission Electron Microscopy Image Sequences
## Overview
High-Resolution Transmission Electron Microscopy (HR-TEM) is a crucial technique in materials science, enabling the visualization of atomic structures with sub-nanometer resolution. However, accurate segmentation and tracking of nanoparticles in HR-TEM images remain challenging due to factors such as noise, contrast variations, and dynamic morphological changes.

This repository contains the codebase for the project **"Semi-Supervised and Temporally-Aware Segmentation of High-Resolution Transmission Electron Microscopy Image Sequences."**  
We implement and evaluate several segmentation models for HR-TEM images, including:
- **SwinTCN-Seg** (our proposed spatiotemporal model)
- **U-Net**
- **U-Net++**
- **FPN**
- **DeepLabV3+**
- **SegFormer**

The framework employs **semi-supervised learning** using **pseudo-labeling** and **self-training** to leverage both labeled and unlabeled data. The final goal is to robustly segment dynamically evolving nanoparticle structures captured in HR-TEM sequences.

## Installation and Setup

1. Clone the repository:

```bash
git clone https://github.com/kaur-manpreet325/TEM-Seg.git
cd TEM-Seg
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # For Linux/macOS
.\venv\Scripts\activate   # For Windows
```

3. Install required dependencies:

```bash
pip install -r requirements.txt
```

Main packages to include in `requirements.txt`:

```
torch
torchvision
segmentation-models-pytorch
albumentations
opencv-python
pandas
scikit-learn
tqdm
monai
```

## How to Run the Code

### 1. Preprocessing (Data Splitting)

Organize your data:
- Raw HR-TEM frames in a folder (e.g., `raw_frames/`)
- Corresponding ground truth masks (e.g., `ground_masks/`)
- Corresponding directory to save split data (e.g., `split_labeled_data/`)
  
Run:

```bash
python Data_Handling.py
```

This will create the `train/`, `valid/`, `test/`, and `unlabeled/` folders.

## 2. YOLO+SAM-based Semi-Automated Annotation

- To generate segmentation masks for HR-TEM frames using a semi-automated pipeline based on YOLOv8 and SAM:

### Step 1: Clone the Repository

- Run:
  ```bash
  git clone https://github.com/ArdaGen/STEM-Automated-Nanoparticle-Analysis-YOLOv8-SAM.git
  cd STEM-Automated-Nanoparticle-Analysis-YOLOv8-SAM

## Step 2: Setup
Follow the instructions in the YOLOv8-SAM repository to:

- Download pretrained weights (YOLOv8 and SAM)

- Configure the environment

- Adjust input/output directory paths as needed in DATA/Raw/YOLO+SAM.py

## Step 3: Run Annotation Pipeline
Use the provided script to perform object detection with YOLOv8 and mask generation with SAM. The pipeline automatically refines bounding boxes into segmentation masks. These generated masks can be saved in an output folder and used as an alternative to manual ground truths (5%) in the SwinTCN-Seg pipeline for self-training.

### 3. Training a Model

Each model has a dedicated script. Examples:

- Train **SwinTCN-Seg**:

```bash
python main.py
```

- Train **baseline models**:

```bash
python UNet.py
python U_Net++.py
python FPN.py
python DeepLabV3+.py
python Segformer.py
```

Each script performs:
- Initial supervised training
- Semi-supervised self-training with pseudo-labels
- Saving model checkpoints and test predictions

## Special Instructions

- **Path Setup:**  
  Update `base_save_dir` and `mask_dir` in each script to match your directory structure:

```python
base_save_dir = "/path/to/split_labeled_data"
mask_dir = "/path/to/ground_masks"
```

- **Hardware:**  
  The scripts will automatically detect and use a GPU if available.

- **Self-Training:**  
  Self-training hyperparameters (confidence threshold, iteration number, lambda weight) are customizable within each script.

## Additional Notes

- Evaluation metrics: **Dice Score**, **IoU**, and **Pixel Accuracy**.
- Data augmentations are used during training to improve generalization.
- Early stopping based on validation loss is implemented.
- Pseudo-labels are saved at each self-training iteration.

## Workflow Overview

### 1. Model Architecture: SwinTCN-Seg
![Architecture](Figures/SwinTCN-Seg1.png)  
*Figure: Detailed architecture of the TCN-SwinUNETR model. This model integrates hierarchical spatial features from Swin Transformer with temporal modeling using a Temporal Convolutional Network for robust HR-TEM segmentation.*

### 2. End-to-End Pipeline
![Workflow](Figures/Semi-Supervised.png)  
*Figure: Workflow of the semi-supervised segmentation pipeline. The top branch illustrates the supervised learning with labeled data, while the bottom shows pseudo-labeling and self-training loop.*
