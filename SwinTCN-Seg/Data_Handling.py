import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil

def normalize_filename(path):
    """Extract normalized filename (without directory and extension)."""
    return os.path.basename(path).replace(' ', '').split('.')[0]

def create_and_save_splits(raw_dir, mask_dir, base_save_dir, test_size=0.2, val_size=0.5, random_state=42):
    raw_files = sorted(glob.glob(os.path.join(raw_dir, 'frame_*.png')))
    mask_files = sorted(glob.glob(os.path.join(mask_dir, 'frame_*.png')))  

    # Create map from normalized mask filenames
    mask_map = {normalize_filename(m): m for m in mask_files}

    labeled_images = []
    labeled_masks = []
    unlabeled_images = []

    for raw in raw_files:
        norm_name = normalize_filename(raw)
        if norm_name in mask_map:
            labeled_images.append(raw)
            labeled_masks.append(mask_map[norm_name])
        else:
            unlabeled_images.append(raw)

    labeled_df = pd.DataFrame({'images': labeled_images, 'masks': labeled_masks})
    unlabeled_df = pd.DataFrame({'images': unlabeled_images, 'masks': [None] * len(unlabeled_images)})

    # Split into train/val/test
    train_df, valid_df = train_test_split(labeled_df, test_size=test_size, random_state=random_state)
    valid_df, test_df = train_test_split(valid_df, test_size=val_size, random_state=random_state)

    # Make folders
    os.makedirs(base_save_dir, exist_ok=True)
    for subset in ['train', 'valid', 'test', 'unlabeled']:
        os.makedirs(os.path.join(base_save_dir, subset), exist_ok=True)

    # Copy raw images
    def copy_raw_images(df, subset):
        for _, row in df.iterrows():
            src_image = row['images']
            dest_image = os.path.join(base_save_dir, subset, os.path.basename(src_image))
            shutil.copy(src_image, dest_image)

    copy_raw_images(train_df, 'train')
    copy_raw_images(valid_df, 'valid')
    copy_raw_images(test_df, 'test')
    copy_raw_images(unlabeled_df, 'unlabeled')

    return train_df, valid_df, test_df, unlabeled_df

# --- Main Execution (Modify paths based on your actual directory structure)---
if __name__ == '__main__':
    raw_paths = [
        "/path/to/raw_frames"
    ]

    mask_paths = [
        "/path/to/ground_masks"
    ]

    for raw_dir, mask_dir in zip(raw_paths, mask_paths):
        dataset_name = os.path.basename(raw_dir.rstrip('/'))
        save_dir = os.path.join("/path/to/split_labeled_data", dataset_name)
        print(f"\n Processing dataset: {dataset_name}")
        try:
            train_df, valid_df, test_df, unlabeled_df = create_and_save_splits(raw_dir, mask_dir, save_dir)
            print(f" Done: {dataset_name}")
            print(f" Train: {len(train_df)} | Valid: {len(valid_df)} | Test: {len(test_df)} | Unlabeled: {len(unlabeled_df)}")
        except Exception as e:
            print(f" Error processing {dataset_name}: {e}")
