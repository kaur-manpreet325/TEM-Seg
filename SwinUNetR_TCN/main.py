import os
import glob
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from monai.networks.nets import SwinUNETR
from torch.nn.utils import weight_norm
import torch.nn.functional as F
import shutil

# Constants
IMG_SIZE = 512
BATCH_SIZE = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 1e-4
EPOCHS = 300
PATIENCE = 25  # Early stopping patience
FEATURE_SIZE = 84 # Feature size extracted from Swin UNETR
TCN_CHANNELS = [128, 256, 512]
NUM_CLASSES = 1  # Segmentation is binary (foreground/background)
INITIAL_UNCERTAINTY_THRESHOLD = 0.9  # Initial threshold for pseudo-label confidence
INITIAL_SELF_TRAIN_RATIO = 0.5 # Initial ratio of unlabeled data to use for self-training
NUM_ENSEMBLE = 5 # Number of ensemble models to use for pseudo labeling
NUM_SELF_TRAINING_ITERATIONS = 5  # Number of times to repeat self-training

# Paths (update these based on your directory structure)
base_save_dir = "/path/to/split_labeled_data"  # Split directories for images
mask_dir = "/path/to/ground_masks"   # Original directory for masks

train_dir = os.path.join(base_save_dir, "train")
valid_dir = os.path.join(base_save_dir, "valid")
test_dir = os.path.join(base_save_dir, "test")
unlabeled_dir = os.path.join(base_save_dir, "unlabeled")

# Helper function to create DataFrame from image and mask directories
def create_dataframe(image_dir, mask_dir=None):
    """
    Creates a DataFrame with image and corresponding mask paths.

    Args:
        image_dir (str): Path to the directory containing images.
        mask_dir (str): Path to the directory containing masks (optional).

    Returns:
        pd.DataFrame: DataFrame with columns 'images' and 'masks'.
    """
    image_files = sorted(glob.glob(os.path.join(image_dir, '*.png')))
    
    if mask_dir:
        # Match masks based on filenames in the image directory
        mask_files = [
            os.path.join(mask_dir, os.path.basename(f)) for f in image_files
            if os.path.exists(os.path.join(mask_dir, os.path.basename(f)))
        ]
        df = pd.DataFrame({
            'images': image_files,
            'masks': mask_files
        })
    else:
        # For unlabeled data, masks are set to None
        df = pd.DataFrame({
            'images': image_files,
            'masks': [None] * len(image_files)
        })
    
    return df

# Create DataFrames for train, validation, test, and unlabeled sets
train_df = create_dataframe(train_dir, mask_dir)  # Use original mask directory for labeled data
valid_df = create_dataframe(valid_dir, mask_dir)
test_df = create_dataframe(test_dir, mask_dir)
unlabeled_df = create_dataframe(unlabeled_dir)  # No masks for unlabeled data

# Augmentations
def get_train_augs():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.GaussNoise(var_limit=(10, 50), p=0.5),
        A.MotionBlur(blur_limit=3, p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=10, p=0.3)
    ], is_check_shapes=False)

def get_valid_augs():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE)
    ], is_check_shapes=False)

def get_test_augs():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE)
    ], is_check_shapes=False)

class SegmentationDataset(Dataset):
    def __init__(self, df, augmentation):
        self.df = df
        self.augmentation = augmentation

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row.images
        mask_path = row.masks

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            h, w = image.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)

        if self.augmentation:
            augmented = self.augmentation(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        image = np.transpose(image, (2, 0, 1)).astype(np.float32) / 255.0
        mask = np.expand_dims(mask, 0).astype(np.float32) / 255.0

        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)

        return torch.tensor(image), torch.tensor(mask.squeeze(0))

# Create datasets and dataloaders
trainset = SegmentationDataset(train_df, get_train_augs())
validset = SegmentationDataset(valid_df, get_valid_augs())
unlabeledset = SegmentationDataset(unlabeled_df, get_valid_augs())

trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
validloader = DataLoader(validset, batch_size=BATCH_SIZE, shuffle=False)
unlabeledloader = DataLoader(unlabeledset, batch_size=BATCH_SIZE, shuffle=False)

testset = SegmentationDataset(test_df, get_test_augs())
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

seq_length = 30

# TCN Components
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.1):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.1):
        super(TemporalConvNet, self).__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCNSwinModel(nn.Module):
    def __init__(self, mc_dropout=False):
        super(TCNSwinModel, self).__init__()

        self.swin_unetr = SwinUNETR(
            img_size=(IMG_SIZE, IMG_SIZE),
            in_channels=3,
            out_channels=FEATURE_SIZE,
            feature_size=FEATURE_SIZE,
            use_checkpoint=False,
            spatial_dims=2,
            use_v2=True
        )

        self.tcn = TemporalConvNet(FEATURE_SIZE, TCN_CHANNELS, kernel_size=2, dropout=0.1)

        self.segmentation_head = nn.Conv2d(TCN_CHANNELS[-1], NUM_CLASSES, kernel_size=1)
        self.mc_dropout = mc_dropout

    def forward(self, data):
        batch_size, time_steps, channels, height, width = data.shape
        spatial_features = torch.zeros((batch_size, time_steps, FEATURE_SIZE, height, width)).to(data.device)

        for t in range(time_steps):
            spatial_features[:, t, :, :, :] = self.swin_unetr(data[:, t, :, :, :])

        spatial_features = spatial_features.view(batch_size, time_steps, FEATURE_SIZE, height * width)
        spatial_features = spatial_features.permute(0, 2, 1, 3).reshape(batch_size, FEATURE_SIZE, time_steps * height * width)

        temporal_features = self.tcn(spatial_features).view(batch_size, TCN_CHANNELS[-1], time_steps, height, width)
        output = self.segmentation_head(torch.sum(temporal_features, dim=2))

        if self.mc_dropout:
            output = F.dropout(output, p=0.5, training=True)

        return output

class SegmentationModel(nn.Module):
    def __init__(self, mc_dropout=False):
        super(SegmentationModel, self).__init__()
        self.model = TCNSwinModel(mc_dropout=mc_dropout)

    def forward(self, images, masks=None):
        logits = self.model(images)
        
        if masks is not None:
            loss1 = smp.losses.DiceLoss(mode='binary')(logits, masks)
            loss2 = nn.BCEWithLogitsLoss()(logits, masks)
            return logits, loss1 + loss2
        return logits

# Metrics
def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1e-7) / (np.sum(y_true_f) + np.sum(y_pred_f) + 1e-7)

def iou_coef(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f) + np.sum(y_pred_f) - intersection
    return (intersection + 1e-7) / (union + 1e-7)

def pixel_accuracy(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    correct = np.sum(y_true_f == y_pred_f)
    total = len(y_true_f)
    return correct / total

def train_fn(data_loader, model, optimizer):
    model.train()
    total_loss = 0.0
    for images, masks in tqdm(data_loader):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)
        optimizer.zero_grad()
        _, loss = model(images, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

def eval_fn(data_loader, model):
    model.eval()
    total_loss = 0.0
    dice_scores = []
    iou_scores = []
    pixel_acc_scores = []
    with torch.no_grad():
        for images, masks in tqdm(data_loader):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            logits, loss = model(images, masks)
            total_loss += loss.item()

            preds = torch.sigmoid(logits).cpu().numpy() > 0.5
            masks = masks.cpu().numpy()

            dice_scores.extend([dice_coef(mask, pred) for mask, pred in zip(masks, preds)])
            iou_scores.extend([iou_coef(mask, pred) for mask, pred in zip(masks, preds)])
            pixel_acc_scores.extend([pixel_accuracy(mask, pred) for mask, pred in zip(masks, preds)])

    return total_loss / len(data_loader), np.mean(dice_scores), np.mean(iou_scores), np.mean(pixel_acc_scores)

# Training loop
def train_model(model, trainloader, validloader, epochs, patience):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_valid_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        train_loss = train_fn(trainloader, model, optimizer)
        valid_loss, valid_dice, valid_iou, valid_pixel_acc = eval_fn(validloader, model)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Valid Loss: {valid_loss:.4f}")

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'best_model.pt')
            print("Saved best model")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping after {epoch+1} epochs")
            break

    print("Training completed.")
    return best_valid_loss

# Initial training
model = SegmentationModel().to(DEVICE)
best_valid_loss = train_model(model, trainloader, validloader, EPOCHS, PATIENCE)

# Function to generate pseudo-labels
def generate_pseudo_labels(model, data_loader, threshold):
    model.eval()
    pseudo_labels = []
    confidences = []
    with torch.no_grad():
        for images, _ in tqdm(data_loader):
            images = images.to(DEVICE)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            pseudo_labels.extend((probs > 0.5).astype(np.uint8))
            confidences.extend(np.max(probs, axis=(1, 2, 3)))
    return np.array(pseudo_labels), np.array(confidences)

# Self-training loop
for iteration in range(NUM_SELF_TRAINING_ITERATIONS):
    print(f"Self-training iteration {iteration + 1}/{NUM_SELF_TRAINING_ITERATIONS}")

    # Generate pseudo-labels
    pseudo_labels, confidences = generate_pseudo_labels(model, unlabeledloader, INITIAL_UNCERTAINTY_THRESHOLD)

    # Dynamically adjust threshold and ratio
    current_threshold = INITIAL_UNCERTAINTY_THRESHOLD + (iteration * 0.02)  # Increase threshold slightly each iteration
    current_ratio = INITIAL_SELF_TRAIN_RATIO - (iteration * 0.1)  # Decrease ratio each iteration
    current_ratio = max(current_ratio, 0.1)  # Ensure ratio doesn't go below 0.1

    # Select high-confidence pseudo-labels
    high_confidence_indices = np.where(confidences > current_threshold)[0]
    num_samples = int(current_ratio * len(unlabeled_df))
    selected_indices = np.random.choice(high_confidence_indices, min(num_samples, len(high_confidence_indices)), replace=False)

    # Create pseudo-labeled dataset
    pseudo_labeled_df = unlabeled_df.iloc[selected_indices].copy()
    pseudo_labeled_df['masks'] = [None] * len(pseudo_labeled_df)

    # Save pseudo-labels
    output_dir = os.path.join(mask_dir, f'TCN_SwinUNetR_output_iteration_{iteration + 1}')
    os.makedirs(output_dir, exist_ok=True)

    for i, (_, row) in enumerate(pseudo_labeled_df.iterrows()):
        image_name = os.path.basename(row.images)
        pseudo_label_path = os.path.join(output_dir, f'pseudo_{image_name}')
        cv2.imwrite(pseudo_label_path, pseudo_labels[selected_indices[i]][0] * 255)
        pseudo_labeled_df.at[pseudo_labeled_df.index[i], 'masks'] = pseudo_label_path

    # Combine original and pseudo-labeled data
    combined_df = pd.concat([train_df, pseudo_labeled_df], ignore_index=True)

    # Create new dataset and dataloader
    combined_trainset = SegmentationDataset(combined_df, get_train_augs())
    combined_trainloader = DataLoader(combined_trainset, batch_size=BATCH_SIZE, shuffle=True)

    # Retrain the model
    model = SegmentationModel().to(DEVICE)
    new_valid_loss = train_model(model, combined_trainloader, validloader, EPOCHS, PATIENCE)

    # If new model performs worse, revert to previous best model
    if new_valid_loss > best_valid_loss:
        print("New model performs worse. Reverting to previous best model.")
        model.load_state_dict(torch.load('best_model.pt'))
    else:
        best_valid_loss = new_valid_loss

# Final evaluation on test set
model.load_state_dict(torch.load('best_model.pt'))
test_loss, test_dice, test_iou, test_pixel_acc = eval_fn(testloader, model)
print(f"Final Test Loss: {test_loss:.4f}")
print(f"Final Test Dice: {test_dice:.4f}")
print(f"Final Test IoU: {test_iou:.4f}")
print(f"Final Test Pixel Accuracy: {test_pixel_acc:.4f}")

# Save predicted masks for test set
test_output_dir = os.path.join(mask_dir, 'TCN_SwinUNetR_test_output')
os.makedirs(test_output_dir, exist_ok=True)

model.eval()
with torch.no_grad():
    for i, (images, _) in enumerate(testloader):
        images = images.to(DEVICE)
        logits = model(images)
        preds = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(np.uint8)
        
        for j, pred in enumerate(preds):
            image_name = os.path.basename(test_df.iloc[i * BATCH_SIZE + j].images)
            output_path = os.path.join(test_output_dir, f'pred_{image_name}')
            cv2.imwrite(output_path, pred[0] * 255)

print(f"Predicted masks for test set saved in {test_output_dir}")
