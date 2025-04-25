import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil

# Paths (modify these based on your actual directory structure)
raw_dir = '/path/to/raw_frames'
mask_dir = '/path/to/ground_masks'
base_save_dir = "/path/to/split_labeled_data"  # Directory to save split data

def create_and_save_splits(raw_dir, mask_dir, base_save_dir, test_size=0.2, val_size=0.5, random_state=42):
    """
    Splits the dataset into training, validation, and test sets, and saves the files
    into separate directories, including unlabeled data.

    Args:
        raw_dir (str): Path to the directory containing raw images.
        mask_dir (str): Path to the directory containing mask images.
        base_save_dir (str): Base directory where train, validation, and test subdirectories will be created.
        test_size (float): Proportion of data to use for the initial test split.
        val_size (float): Proportion of the initial test split to use for validation.
        random_state (int): Random seed for reproducibility.
    """

    # Create DataFrame
    raw_files = sorted(glob.glob(os.path.join(raw_dir, 'frame_*.png')))
    mask_files = sorted(glob.glob(os.path.join(mask_dir, 'frame_*.png')))

    labeled_df = pd.DataFrame({
        'images': [f for f in raw_files if os.path.basename(f) in [os.path.basename(m) for m in mask_files]],
        'masks': [m for m in mask_files if os.path.basename(m) in [os.path.basename(f) for f in raw_files]]
    })

    unlabeled_df = pd.DataFrame({
        'images': [f for f in raw_files if os.path.basename(f) not in [os.path.basename(m) for m in mask_files]],
        'masks': [None] * len([f for f in raw_files if os.path.basename(f) not in [os.path.basename(m) for m in mask_files]])
    })

    # Split labeled data into train, validation, and test sets
    train_df, valid_df = train_test_split(labeled_df, test_size=test_size, random_state=random_state)
    valid_df, test_df = train_test_split(valid_df, test_size=val_size, random_state=random_state)

    os.makedirs(base_save_dir, exist_ok=True)

    # Create subdirectories
    for subset in ['train', 'valid', 'test', 'unlabeled']:
        os.makedirs(os.path.join(base_save_dir, subset), exist_ok=True)

    # Function to copy files
    def copy_files(df, subset):
        for _, row in df.iterrows():
            src_image = row['images']
            dst_image = os.path.join(base_save_dir, subset, os.path.basename(src_image))
            shutil.copy(src_image, dst_image)
            
            if row['masks'] is not None:
                src_mask = row['masks']
                dst_mask = os.path.join(base_save_dir, subset, os.path.basename(src_mask))
                shutil.copy(src_mask, dst_mask)

    # Copy files to respective directories
    copy_files(train_df, 'train')
    copy_files(valid_df, 'valid')
    copy_files(test_df, 'test')
    copy_files(unlabeled_df, 'unlabeled')

    return train_df, valid_df, test_df, unlabeled_df


if __name__ == '__main__':
    train_df, valid_df, test_df, unlabeled_df = create_and_save_splits(raw_dir, mask_dir, base_save_dir)
    print("Data splitting and saving complete.")
    print(f"Train set size: {len(train_df)}")
    print(f"Validation set size: {len(valid_df)}")
    print(f"Test set size: {len(test_df)}")
    print(f"Unlabeled set size: {len(unlabeled_df)}")
