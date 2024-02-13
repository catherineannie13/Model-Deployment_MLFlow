import os
import shutil
from sklearn.model_selection import train_test_split

def split_data(source_dir, target_dir, test_size=0.2, val_size=0.2):
    """
    Split dataset into training, validation, and test sets.
    :param source_dir: Directory where the class folders (H1, H2, H3, H5, H6) are located.
    :param target_dir: Directory where train, test, and val folders are located.
    :param test_size: Proportion of the dataset to include in the test split.
    :param val_size: Proportion of the training set to include in the validation set.
    """
    classes = ['H1', 'H2', 'H3', 'H5', 'H6']  # Specify your class names here

    for class_name in classes:
        class_dir = os.path.join(source_dir, class_name)
        images = [img for img in os.listdir(class_dir) if img.endswith('.jpg')]
        
        # Splitting dataset
        train_val, test = train_test_split(images, test_size=test_size, random_state=42)
        train, val = train_test_split(train_val, test_size=val_size, random_state=42)
        
        # Function to copy images to respective directories
        def copy_images(images, dest_subdir):
            dest_dir = os.path.join(target_dir, dest_subdir, class_name)
            os.makedirs(dest_dir, exist_ok=True)
            for img_name in images:
                src_path = os.path.join(class_dir, img_name)
                dst_path = os.path.join(dest_dir, img_name)
                shutil.copy(src_path, dst_path)
        
        # Copying images to respective directories
        copy_images(train, 'train')
        copy_images(val, 'val')
        copy_images(test, 'test')