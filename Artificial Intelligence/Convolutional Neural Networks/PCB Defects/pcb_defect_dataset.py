
#! Custom TODO notes:
#TODO ATH: Add type hint.
#TODO CTH: Check type hint.

import cv2
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image
import torch
import torchvision.transforms as T

from xml.etree.ElementTree import Element
import xml.etree.ElementTree as ET


class PCBDefectDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            image_paths: list[str],
            annotation_paths: list[str],
            transforms: T.Compose 
    ) -> None:
        """
        A custom PyTorch Dataset for loading PCB defect images and their annotations.

        This class parses XML annotation files in PASCAL VOC format to extract
        bounding boxes and class labels for object detection tasks.

        Args:
            image_paths (list[str]): A list of file paths to the images.
            annotation_paths (list[str]): A list of file paths to the XML annotations.
            transforms (T.Compose): A transformation pipeline to be applied to the image and target. (Default = None)
        """
        self.image_paths: list[str] = image_paths
        self.annotation_paths: list[str] = annotation_paths
        self.transforms: T.Compose  = transforms 

        # Map the defect types to integer ID's.
        self.defect_type_to_id: dict[str, int] = {
            "missing_hole" : 1,
            "mouse_bite" : 2,
            "open_circuit" : 3,
            "short" : 4,
            "spur" : 5,
            "spurious_copper" : 6
        }

        # Pre-parse the annotations once.
        self.targets: list[dict[str, list]] = [] 
        for ann_pth in self.annotation_paths:
            # Parse the XML file into a tree structure.
            tree: ET = ET.parse(ann_pth) 
            # Get the top level element (a.k.a the 'root'). In this case the <annotation> tag.
            root: Element = tree.getroot() 
            
            boxes: list[list[int, int, int, int]] = []
            labels: list[int] = []

            # Find all <object> tags in the file.
            for member in root.findall("object"):
                # Inside an <object>, find the <name> tag.
                defect_type: str = member.find("name").text
                # Map the found defect name into an integer ID.
                labels.append(self.defect_type_to_id[defect_type])

                # Inside an <object>, find the <bndbox> tag.
                bndbox: Element = member.find("bndbox") 
                # Inside the <bndbox>, find each coordinate and convert it to an integer.
                xmin: int = int(bndbox.find("xmin").text)
                ymin: int = int(bndbox.find("ymin").text)
                xmax: int = int(bndbox.find("xmax").text)
                ymax: int = int(bndbox.find("ymax").text)
                boxes.append([xmin, ymin, xmax, ymax])
            self.targets.append(
                {
                    "boxes": boxes,
                    "labels": labels
                }
            )





    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        # Return the total number of samples.
        return len(self.image_paths)

    def __getitem__(
            self,
            idx: int
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Loads an image and its annotations, applies transformations, and returns them.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            A tuple containing the transformed image and a target dictionary. The
            target dictionary contains the 'boxes', 'labels', 'image_id', and
            'area' as torch.Tensors.
        """
        # Since the DataSet is only responsible for a single data in the dataset,
        #   we only transform a specific indexed one. DataLoader creates the batches.
        image_path: str = self.image_paths[idx]
        # Open the image and convert into RGB numpy.
        image = Image.open(image_path).convert("RGB")
        image_np: NDArray = np.array(image)

        # Create an shallow copy for the target dictionary.
        target_shallow = dict(self.targets[idx])
        # If needed, apply the transformation.
        if self.transforms:
            # Pass data as keyword arguments.
            transformed: dict[str, Any] = self.transforms(
                image = image_np,
                bboxes = target_shallow["boxes"],
                labels = target_shallow["labels"]
            )
            # Albumentations return a dictionary.
            image: NDArray = transformed["image"]
            # In case the image does not have any bounding boxes(so no defects), we should use .get()
            boxes: list[list[float]] = transformed.get("boxes", []) # If the key is missing, it returns a default empty list.
            labels: list[int] = transformed.get("labels", [])
            # Update the transformed image with new info of bounding boxes and labels.
            if not boxes: # If there are not bounded boxes, make sure that the shape is still [0, 4].
                boxes: torch.Tensor = torch.zeros((0, 4), dtype=torch.float32)
            else:
                boxes: torch.Tensor = torch.as_tensor(boxes, dtype=torch.float32)
            target_shallow["boxes"] = boxes
            target_shallow["labels"] = torch.as_tensor(labels, dtype=torch.int64)
            target_shallow["image_id"] = torch.tensor([idx])

            # If the box tensor is not empty, calculate the area for each box.
            if boxes.shape[0] > 0:
                area: torch.Tensor = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            else: # Handle the case without any objects.
                area = torch.as_tensor([], dtype=torch.float32)
            target_shallow["area"] = area

        return image, target_shallow

