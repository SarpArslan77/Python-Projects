
#TODO Change the transformation pipeline to using albumentations library.

#! Custom TODO notes:
#TODO AD: Add docstring.
#TODO ATH: Add type hint.
#TODO CTH: Check type hint.
#TODO FTH: Fix the hardcoding.
#TODO AA: Add assertation.

import glob
import os

import albumentations as A
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

from pcb_defect_dataset import PCBDefectDataset

def pcb_defect_collate_fn(batch: list[tuple[torch.Tensor, dict]]) -> tuple[torch.Tensor, tuple[dict]]: #TODO ATH
    """
    A custom collate function for the PCB Defect object detection dataset.

    This function is passed to the DataLoader to handle the creation of batches. 
    It takes a list of samples (each a tuple of an image tensor and a target dict),
    stacks the image tensors, and returns the targets as a tuple of dictionaries. 
    This is necessary because the target dictionaries can have a variable number
    of annotations per image, which prevents them from being stacked into a single tensor. 
    Which is how the normal collate function of dataloader works.

    Args:
        batch (list[tuple[torch.Tensor, tuple[dict]]]): A list of tuples, where each tuple is the output of PCBDefectDataset's __getitem__ method.

    Returns:
        tuple[torch.Tensor, tuple[dict]]: Contains a tensor of stacked images and a tuple of annotation dictionaries. 
    """
    images, annotations = zip(*batch)

    images_tensor: torch.Tensor = torch.stack(images)

    return images_tensor, annotations

def create_dataloaders( 
        parent_folder: str,
        images_folder_name: str,
        images_format: str,
        annotations_folder_name: str,
        random_state_seed: int,
        batch_size: int,
        num_workers: int
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Gathers image and annotation paths, splits them into train/val/test sets,
    and creates the corresponding PyTorch DataLoaders.

    This function handles the entire data preparation pipeline, from finding
    files to creating batched data loaders ready for training and evaluation.

    Args:
        parent_folder (str): The root directory of the dataset containing the image and annotation folders.
        images_folder_name (str): The name of the folder containing the images.
        images_format (str): The file extension of the images (e.g., "jpg").
        annotations_folder_name (str): The name of the folder containing the XML annotation files.
        random_state_seed (int): A seed for the random number generator to ensure reproducible data splits.
        batch_size (int): The number of samples per batch in each DataLoader.
        num_workers (int): The number of subprocesses to use for data loading.

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: Containing the train_loader, val_loader, and test_loader.
    """

    # Check whether the parent folder exists.
    if not os.path.isdir(parent_folder):
        raise FileNotFoundError(f"The specified parent folder does not exist at: {parent_folder} .")
    # Check whether the batch size is a positive integer.
    if batch_size <= 0:
        raise ValueError(f"batch_size must be a positive integer, but got {batch_size} .")
    # Check whether the number of workers is non negative integer.
    if num_workers < 0:
        raise ValueError(f"num_workers must be a non-negative integer, but got {num_workers} .")

    # Construct the pattern for the recursive search.
    images_folder_path: str = os.path.join(parent_folder, f"{images_folder_name}/") 
    image_pattern: str = os.path.join(
        images_folder_path,
        "**",
        f"*.{images_format}" 
    )
    # Perform the recursive search.
    image_paths: list[str] = glob.glob(
        pathname = image_pattern,
        recursive = True
    ) 
    # Sort the list, ensuring the correct pairing with the annotations.
    image_paths.sort()
    # Repeat the same search for annotation paths.
    annotations_folder_path: str = os.path.join(parent_folder, f"{annotations_folder_name}/")
    annotation_pattern: str = os.path.join(
        annotations_folder_path,
        "**",
        "*.xml" #TODO AA
    )
    annotation_paths: list[str] = glob.glob(
        pathname = annotation_pattern,
        recursive = True
    )
    annotation_paths.sort()
    
    # Assert, that images do exist.
    assert len(image_paths) > 0, \
    f"No images were found in {images_folder_path}"
    # Assert, that every image has a corresponding annotation.
    assert len(image_paths) == len(annotation_paths), \
    f"Number of images does not match the number of annotations. {len(image_paths)} images but {len(annotation_paths)} annotations."

    # Define a transformation pipeline.
    standard_transformation_pipeline = A.Compose(
        [
            
        ]
    )
    #TODO: Add a augmentation transformation pipeline.

    # Split the data into training (70%), validation (%15) and test(%15).
    X_train, X_val_and_test, y_train, y_val_and_test = train_test_split(
        image_paths,
        annotation_paths,
        test_size = 0.3,
        random_state = random_state_seed  # Ensures, that the data is split in the exact same way every time.
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_val_and_test,
        y_val_and_test,
        test_size = 0.5,
        random_state = random_state_seed
    )

    # Create the datasets.
    train_dataset = PCBDefectDataset(
        image_paths = X_train,
        annotation_paths = y_train,
        transforms = standard_transformation_pipeline
    )
    val_dataset = PCBDefectDataset(
        image_paths = X_val,
        annotation_paths = y_val,
        transforms = standard_transformation_pipeline
    )
    test_dataset = PCBDefectDataset(
        image_paths = X_test,
        annotation_paths = y_test,
        transforms = standard_transformation_pipeline
    )

    # Create the dataloaders.
    train_loader = DataLoader(
        dataset = train_dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = num_workers, 
        collate_fn = pcb_defect_collate_fn,
    )
    val_loader = DataLoader(
        dataset = val_dataset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = num_workers, 
        collate_fn = pcb_defect_collate_fn,
    )
    test_loader = DataLoader(
        dataset = test_dataset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = num_workers, 
        collate_fn = pcb_defect_collate_fn,
    )

    return train_loader, val_loader, test_loader


