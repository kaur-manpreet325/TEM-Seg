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
import torch.nn.functional as F
import shutil

# Constants
IMG_SIZE = 512
BATCH_SIZE = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENCODER = 'resnet152'
WEIGHTS = None
LR = 1e-4
EPOCHS = 300
PATIENCE = 25  # Early stopping patience
NUM_CLASSES = 1  # Segmentation is binary (foreground/background)
INITIAL_UNCERTAINTY_THRESHOLD = 0.9  # Initial threshold for pseudo-label confidence
INITIAL_SELF_TRAIN_RATIO = 0.5 # Initial ratio of unlabeled data to use for self-training
NUM_ENSEMBLE = 5 # Number of ensemble models to use for pseudo labeling
NUM_SELF_TRAINING_ITERATIONS = 5  # Number of times to repeat self-training
INITIAL_LAMBDA = 0.1

# Paths (update these based on your directory structure)
base_save_dir = "/path/to/split_labeled_data"  # Split directories for images
mask_dir = "/path/to/ground_masks"   # Original directory for masks

train_dir = os.path.join(base_save_dir, "train")
valid_dir = os.path.join(base_save_dir, "valid")
test_dir = os.path.join(base_save_dir, "test")
unlabeled_dir = os.path.join(base_save_dir, "unlabeled")

# Helper function to create DataFrame from image and mask directories
def create_dataframe(image_dir, mask_dir=None, is_pseudo=False):
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
            'masks': mask_files,
            'is_pseudo': [is_pseudo] * len(image_files)
        })
    else:
        # For unlabeled data, masks are set to None
        df = pd.DataFrame({
            'images': image_files,
            'masks': [None] * len(image_files),
            'is_pseudo': [is_pseudo] * len(image_files)
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

# Dataset class
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
        is_pseudo = row.is_pseudo

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

        return torch.tensor(image), torch.tensor(mask), torch.tensor(is_pseudo)
        
# Create datasets and dataloaders
trainset = SegmentationDataset(train_df, get_train_augs())
validset = SegmentationDataset(valid_df, get_valid_augs())
unlabeledset = SegmentationDataset(unlabeled_df, get_valid_augs())

trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
validloader = DataLoader(validset, batch_size=BATCH_SIZE, shuffle=False)
unlabeledloader = DataLoader(unlabeledset, batch_size=BATCH_SIZE, shuffle=False)

testset = SegmentationDataset(test_df, get_test_augs())
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

class SegmentationModel(nn.Module):
    def __init__(self):
        super(SegmentationModel, self).__init__()
        self.deeplab = smp.FPN(
            encoder_name=ENCODER,
            encoder_weights=WEIGHTS,
            in_channels=3,
            classes=1,
        )

    def forward(self, images, masks=None):
        logits = self.deeplab(images)
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

def train_fn(data_loader, model, optimizer, lambda_factor):
    model.train()
    total_loss = 0.0
    for images, masks, is_pseudo in tqdm(data_loader):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)
        is_pseudo = is_pseudo.to(DEVICE).float()

        optimizer.zero_grad()
        logits = model(images)
        dice_loss = smp.losses.DiceLoss(mode='binary')(logits, masks)
        bce_loss = F.binary_cross_entropy_with_logits(logits, masks)
        loss_per_sample = dice_loss + bce_loss

        supervised_loss = (loss_per_sample * (1 - is_pseudo)).mean()
        pseudo_loss = (loss_per_sample * is_pseudo).mean()
        loss = supervised_loss + lambda_factor * pseudo_loss

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
        for images, masks, _ in tqdm(data_loader):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            logits = model(images)
            loss1 = smp.losses.DiceLoss(mode='binary')(logits, masks)
            loss2 = nn.BCEWithLogitsLoss()(logits, masks)
            loss = loss1 + loss2
            total_loss += loss.item()

            preds = torch.sigmoid(logits).cpu().numpy() > 0.5
            masks = masks.cpu().numpy()

            dice_scores.extend([dice_coef(mask, pred) for mask, pred in zip(masks, preds)])
            iou_scores.extend([iou_coef(mask, pred) for mask, pred in zip(masks, preds)])
            pixel_acc_scores.extend([pixel_accuracy(mask, pred) for mask, pred in zip(masks, preds)])

    return total_loss / len(data_loader), np.mean(dice_scores), np.mean(iou_scores), np.mean(pixel_acc_scores)


# Training loop
def train_model(model, trainloader, validloader, epochs, patience, lambda_factor):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_valid_loss = float('inf')
    patience_counter = 0
    for epoch in range(epochs):
        train_loss = train_fn(trainloader, model, optimizer, lambda_factor)
        valid_loss, valid_dice, valid_iou, valid_pixel_acc = eval_fn(validloader, model)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Valid Loss: {valid_loss:.4f}")

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'best_model_FPN.pt')
            print("Saved best model")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping after {epoch + 1} epochs")
            break

    print("Training completed.")
    return best_valid_loss


# Function to generate pseudo-labels
def generate_pseudo_labels(model, data_loader, threshold):
    model.eval()
    pseudo_labels = []
    confidences = []
    with torch.no_grad():
        for images, _, _ in tqdm(data_loader):
            images = images.to(DEVICE)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            pseudo_labels.extend((probs > 0.5).astype(np.uint8))
            confidences.extend(np.max(probs, axis=(1, 2, 3)))
    return np.array(pseudo_labels), np.array(confidences)


# Self-training loop
model = SegmentationModel().to(DEVICE)
best_valid_loss = train_model(model, trainloader, validloader, EPOCHS, PATIENCE, 0.0)  # Train initially without lambda

for iteration in range(NUM_SELF_TRAINING_ITERATIONS):
    print(f"Self-training iteration {iteration + 1}/{NUM_SELF_TRAINING_ITERATIONS}")

    # Generate pseudo-labels
    pseudo_labels, confidences = generate_pseudo_labels(model, unlabeledloader,
                                                        INITIAL_UNCERTAINTY_THRESHOLD)

    # Dynamically adjust threshold and ratio
    current_threshold = INITIAL_UNCERTAINTY_THRESHOLD + (iteration * 0.02)  # Increase threshold slightly each iteration
    current_ratio = INITIAL_SELF_TRAIN_RATIO - (iteration * 0.1)  # Decrease ratio each iteration
    current_ratio = max(current_ratio, 0.1)  # Ensure ratio doesn't go below 0.1
    current_lambda = min(INITIAL_LAMBDA + iteration * 0.1, 1.0) 

    # Select high-confidence pseudo-labels
    high_confidence_indices = np.where(confidences > current_threshold)[0]
    num_samples = int(current_ratio * len(unlabeled_df))
    selected_indices = np.random.choice(high_confidence_indices, min(num_samples, len(high_confidence_indices)),
                                         replace=False)

    # Create pseudo-labeled dataset
    pseudo_labeled_df = unlabeled_df.iloc[selected_indices].copy()
    pseudo_labeled_df['masks'] = [None] * len(pseudo_labeled_df)
    pseudo_labeled_df['is_pseudo'] = True  # Mark these as pseudo-labeled

    # Save pseudo-labels
    output_dir = os.path.join(mask_dir, f'FPN_output_iteration_{iteration + 1}')
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
    combined_trainloader = DataLoader(combined_trainset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # Retrain the model
    model = SegmentationModel().to(DEVICE)
    new_valid_loss = train_model(model, combined_trainloader, validloader, EPOCHS, PATIENCE, current_lambda)

    # If new model performs worse, revert to previous best model
    if new_valid_loss > best_valid_loss:
        print("New model performs worse. Reverting to previous best model.")
        model.load_state_dict(torch.load('best_model_FPN.pt'))
    else:
        best_valid_loss = new_valid_loss

# Final evaluation on test set
model.load_state_dict(torch.load('best_model_FPN.pt'))
test_loss, test_dice, test_iou, test_pixel_acc = eval_fn(testloader, model)
print(f"Final Test Loss: {test_loss:.4f}")
print(f"Final Test Dice: {test_dice:.4f}")
print(f"Final Test IoU: {test_iou:.4f}")
print(f"Final Test Pixel Accuracy: {test_pixel_acc:.4f}")

# Save predicted masks for test set
test_output_dir = os.path.join(mask_dir, 'FPN_test_output')
os.makedirs(test_output_dir, exist_ok=True)

model.eval()
with torch.no_grad():
    for i, (images, _, _) in enumerate(testloader):
        images = images.to(DEVICE)
        logits = model(images)
        preds = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(np.uint8)

        for j, pred in enumerate(preds):
            image_name = os.path.basename(test_df.iloc[i * BATCH_SIZE + j].images)
            output_path = os.path.join(test_output_dir, f'pred_{image_name}')
            cv2.imwrite(output_path, pred[0] * 255)

print(f"Predicted masks for test set saved in {test_output_dir}")
