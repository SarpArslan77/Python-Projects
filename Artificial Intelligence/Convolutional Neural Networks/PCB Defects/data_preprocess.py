
#! Custom TODO notes:
#TODO AD: Add docstring.
#TODO ATH: Add type hint.
#TODO CTH: Check type hint.
#TODO FTH: Fix the hardcoding.
#TODO AA: Add assertation.

import cv2
import glob
import os
from typing import Any

import albumentations as A
from albumentations.pytorch import ToTensorV2
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from pcb_defect_dataset import PCBDefectDataset

def preload_images(
        image_paths: list[str],
) -> list[NDArray]:
    """
    Loads all images from a list of paths into RAM.

    Args:
        image_paths (list[str]): A sorted list of file paths to the images.

    Returns:
        list[np.ndarray]: A list of images, where each image is a NumPy array in RGB format.
    """

    preloaded_images: list[NDArray] = []
    print("\nPreloading images into RAM...")
    for image_path in tqdm(image_paths, desc="Preloading Images", unit=" img"):
        # Read the image using OpenCV.
        image: NDArray = cv2.imread(image_path)
        # Convert from BGR to RGB to match the standard formats.
        image_rgb: NDArray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        preloaded_images.append(image_rgb)
    
    print("Image preloading complete!")
    return preloaded_images

def pcb_defect_collate_fn(batch: list[tuple[torch.Tensor, dict]]) -> tuple[torch.Tensor, tuple[dict]]: 
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
        parent_folder_path: str,
        images_folder_name: str,
        images_format: str,
        annotations_folder_name: str,
        device: torch.device,
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
    if not os.path.isdir(parent_folder_path):
        raise FileNotFoundError(f"The specified parent folder does not exist at: {parent_folder_path} .")
    # Check whether the batch size is a positive integer.
    if batch_size <= 0:
        raise ValueError(f"batch_size must be a positive integer, but got {batch_size} .")
    # Check whether the number of workers is non negative integer.
    if num_workers < 0:
        raise ValueError(f"num_workers must be a non-negative integer, but got {num_workers} .")

    # Construct the pattern for the recursive search.
    images_folder_path: str = os.path.join(parent_folder_path, f"{images_folder_name}/") 
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
    annotations_folder_path: str = os.path.join(parent_folder_path, f"{annotations_folder_name}/")
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
            A.Resize(
                height = 600, #TODO FTH
                width = 600,  #TODO FTH
                interpolation = cv2.INTER_LINEAR, #TODO FTH
            ),
            A.Normalize(
                mean = [0.5, 0.5, 0.5], #TODO: Replace with dataset's actual mean.
                std = [0.5, 0.5, 0.5] #TODO Replace with dataset's actual standard deviation.
            ),
            ToTensorV2()
        ],
        bbox_params = {
            "format": "pascal_voc", # [x_min, y_min, x_max, y_max]
            "label_fields": ["labels"]
        }
    )
    #TODO: Add a augmentation transformation pipeline.

    # Preload all images into memory using the paths.
    all_images_preloaded: list[NDArray] = preload_images(image_paths)

    # Split the data into training (70%), validation (%15) and test(%15).
    X_train, X_val_and_test, y_train, y_val_and_test = train_test_split(
        all_images_preloaded,
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

    # Define common keyword arguments about multiprocessing for the dataloaders.
    loader_kwargs: dict[str, Any] = {
        "num_workers": 0,
        "pin_memory": False, # False = CPU and True = GPU.
        "persistent_workers": False, # Keep workers alive between epochs. 
        "prefetch_factor": None # Number of batches loader ahead per worker.
    }
    # Add multiprocessing-specific arguments, only if 'num_workers' > 0.
    if num_workers > 0:
        loader_kwargs.update(
            {
                "num_workers": num_workers,
                "persistent_workers": True,
                "prefetch_factor": 2 #TODO FTH
            }
        )
    # Add GPU-spesific arguments, only if 'device' is not cpu.
    if device.type != "cpu":
        loader_kwargs.update(
            {
                "pin_memory": True
            }
        )

    # Create the datasets.
    train_dataset = PCBDefectDataset(
        preloaded_images = X_train,
        annotation_paths = y_train,
        transforms = standard_transformation_pipeline
    )

    val_dataset = PCBDefectDataset(
        preloaded_images = X_val,
        annotation_paths = y_val,
        transforms = standard_transformation_pipeline
    )
    test_dataset = PCBDefectDataset(
        preloaded_images = X_test,
        annotation_paths = y_test,
        transforms = standard_transformation_pipeline
    )

    # Create the dataloaders.
    train_loader = DataLoader(
        dataset = train_dataset,
        batch_size = batch_size,
        shuffle = True,
        collate_fn = pcb_defect_collate_fn,
        **loader_kwargs
    )
    val_loader = DataLoader(
        dataset = val_dataset,
        batch_size = batch_size,
        shuffle = False,
        collate_fn = pcb_defect_collate_fn,
        **loader_kwargs
    )
    test_loader = DataLoader(
        dataset = test_dataset,
        batch_size = batch_size,
        shuffle = False,
        collate_fn = pcb_defect_collate_fn,
        **loader_kwargs
    )

    return train_loader, val_loader, test_loader


