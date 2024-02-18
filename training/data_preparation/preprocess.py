import os
import shutil
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def split_data(source_dir, target_dir, test_size=0.2, val_size=0.2):
    """
    Split dataset into training, validation, and test sets.

    Parameters:
    - source_dir (str): Directory where the class folders (H1, H2, H3, H5, H6) are located.
    - target_dir (str): Directory where train, test, and val folders are located.
    - test_size (float): Proportion of the dataset to include in the test split.
    - val_size (float): Proportion of the training set to include in the validation set.

    Returns:
    - None
    """
    classes = ['H1', 'H2', 'H3', 'H5', 'H6'] 

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


def create_data_generators(train_dir, val_dir, test_dir, image_size, batch_size):
    """
    Create data generators for training, validation, and test datasets.

    Parameters:
    - train_dir (str): Path to the training dataset directory.
    - val_dir (str): Path to the validation dataset directory.
    - test_dir (str): Path to the test dataset directory.
    - image_size (tuple of int): The target size of the images (width, height).
    - batch_size (int): The size of the batches of data.

    Returns:
    - train_generator (ImageDataGenerator): Data generator for the training dataset.
    - val_generator (ImageDataGenerator): Data generator for the validation dataset.
    - test_generator (ImageDataGenerator): Data generator for the test dataset.
    """
    
    # Preprocessing and realtime data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,  # Rescale pixel values to [0, 1]
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    test_val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    val_generator = test_val_datagen.flow_from_directory(
        val_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_generator = test_val_datagen.flow_from_directory(
        test_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False 
    )

    return train_generator, val_generator, test_generator