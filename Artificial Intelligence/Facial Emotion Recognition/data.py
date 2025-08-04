
#TODO turn the data into tensor and y_test dataloader to be imported in the main file
#TODO: first stack them then cat them!

import os
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
import torch

# Turn the images in the folders to numpy arrays for LLM Model training
training_read_file_path: str = r"C:/Users/Besitzer/Desktop/Python/AI Projects/Facial Emotion Recognition/train"
test_read_file_path: str = r"C:/Users/Besitzer/Desktop/Python/AI Projects/Facial Emotion Recognition/test"

def images_to_tensor(folder_path: str):
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".jpg")]
    
    images: list = []

    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        img = Image.open(img_path)
        img_array = np.array(img)
        img_tensor = torch.from_numpy(img_array)
        images.append(img_tensor)

    return images

batch_size: int = 25

# Turn the test images to tensors and stack them.
test_angry_images = images_to_tensor((test_read_file_path + "/angry"))
test_disgust_images = images_to_tensor((test_read_file_path + "/disgust"))
test_fear_images = images_to_tensor((test_read_file_path + "/fear"))
test_happy_images = images_to_tensor((test_read_file_path + "/happy"))
test_neutral_images = images_to_tensor((test_read_file_path + "/neutral"))
test_sad_images = images_to_tensor((test_read_file_path + "/sad"))
test_surprise_images = images_to_tensor((test_read_file_path + "/surprise"))

test_angry_images_stacked = torch.stack(test_angry_images, dim=0)
test_disgust_images_stacked = torch.stack(test_disgust_images, dim=0)
test_fear_images_stacked = torch.stack(test_fear_images, dim=0)
test_happy_images_stacked = torch.stack(test_happy_images, dim=0)
test_neutral_images_stacked = torch.stack(test_neutral_images, dim=0)
test_sad_images_stacked = torch.stack(test_sad_images, dim=0)
test_surprise_images_stacked = torch.stack(test_surprise_images, dim=0)
# Concate them all stacked emotions to one big stack.
X_test = torch.cat(
    [
        test_angry_images_stacked,
        test_disgust_images_stacked,
        test_fear_images_stacked,
        test_happy_images_stacked,
        test_neutral_images_stacked,
        test_sad_images_stacked,
        test_surprise_images_stacked
    ],
    dim=0
)
# Turn uint8 (Byte) into float32 (Float).
X_test = X_test.float() / 255.0 

# Create labels for emotions.
test_num_angry: int = test_angry_images_stacked.shape[0]
test_num_disgust: int = test_disgust_images_stacked.shape[0]
test_num_fear: int = test_fear_images_stacked.shape[0]
test_num_happy: int = test_happy_images_stacked.shape[0]
test_num_neutral: int = test_neutral_images_stacked.shape[0]
test_num_sad: int = test_sad_images_stacked.shape[0]
test_num_surprise: int = test_surprise_images_stacked.shape[0]
# Create label tensors.
test_angry_labels = torch.full((test_num_angry,), 1, dtype=torch.long)
test_disgust_labels = torch.full((test_num_disgust,), 1, dtype=torch.long)
test_fear_labels = torch.full((test_num_fear,), 1, dtype=torch.long)
test_happy_labels = torch.full((test_num_happy,), 1, dtype=torch.long)
test_neutral_labels = torch.full((test_num_neutral,), 1, dtype=torch.long)
test_sad_labels = torch.full((test_num_sad,), 1, dtype=torch.long)
test_surprise_labels = torch.full((test_num_surprise,), 1, dtype=torch.long)
# Concate them together.
all_test_labels: list = [
    test_angry_labels, test_disgust_labels, test_fear_labels,
    test_happy_labels, test_neutral_labels, test_sad_labels,
    test_surprise_labels
]
y_test = torch.cat(all_test_labels, dim=0)

# Create DataLoader compatible datasets.
test_dataset = TensorDataset(X_test, y_test)

# Use DataLoader for easer iterating over the sets.
test_loader = DataLoader(
    dataset = test_dataset,
    batch_size = batch_size,
    shuffle = False
)

# Repeat the same steps for training data.
training_angry_images = images_to_tensor((training_read_file_path + "/angry"))
training_disgust_images = images_to_tensor((training_read_file_path + "/disgust"))
training_fear_images = images_to_tensor((training_read_file_path + "/fear"))
training_happy_images = images_to_tensor((training_read_file_path + "/happy"))
training_neutral_images = images_to_tensor((training_read_file_path + "/neutral"))
training_sad_images = images_to_tensor((training_read_file_path + "/sad"))
training_surprise_images = images_to_tensor((training_read_file_path + "/surprise"))

training_angry_images_stacked = torch.stack(training_angry_images, dim=0)
training_disgust_images_stacked = torch.stack(training_disgust_images, dim=0)
training_fear_images_stacked = torch.stack(training_fear_images, dim=0)
training_happy_images_stacked = torch.stack(training_happy_images, dim=0)
training_neutral_images_stacked = torch.stack(training_neutral_images, dim=0)
training_sad_images_stacked = torch.stack(training_sad_images, dim=0)
training_surprise_images_stacked = torch.stack(training_surprise_images, dim=0)

X_training = torch.cat(
    [
        training_angry_images_stacked,
        training_disgust_images_stacked,
        training_fear_images_stacked,
        training_happy_images_stacked,
        training_neutral_images_stacked,
        training_sad_images_stacked,
        training_surprise_images_stacked
    ],
    dim=0
)

X_training = X_training.float() / 255.0

training_num_angry: int = training_angry_images_stacked.shape[0]
training_num_disgust: int = training_disgust_images_stacked.shape[0]
training_num_fear: int = training_fear_images_stacked.shape[0]
training_num_happy: int = training_happy_images_stacked.shape[0]
training_num_neutral: int = training_neutral_images_stacked.shape[0]
training_num_sad: int = training_sad_images_stacked.shape[0]
training_num_surprise: int = training_surprise_images_stacked.shape[0]

training_angry_labels = torch.full((training_num_angry,), 1, dtype=torch.long)
training_disgust_labels = torch.full((training_num_disgust,), 1, dtype=torch.long)
training_fear_labels = torch.full((training_num_fear,), 1, dtype=torch.long)
training_happy_labels = torch.full((training_num_happy,), 1, dtype=torch.long)
training_neutral_labels = torch.full((training_num_neutral,), 1, dtype=torch.long)
training_sad_labels = torch.full((training_num_sad,), 1, dtype=torch.long)
training_surprise_labels = torch.full((training_num_surprise,), 1, dtype=torch.long)
all_training_labels: list = [
    training_angry_labels, training_disgust_labels, training_fear_labels,
    training_happy_labels, training_neutral_labels, training_sad_labels,
    training_surprise_labels
]

y_training = torch.cat(all_training_labels, dim=0)

training_dataset = TensorDataset(X_training, y_training)

training_loader = DataLoader(
    dataset = training_dataset,
    batch_size = batch_size,
    shuffle = True
)

print("\nAll data's are prepared and bundled into test- and training_loaders.")