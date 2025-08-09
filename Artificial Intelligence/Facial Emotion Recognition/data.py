
import os
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import torch
import torchvision.transforms as T
from collections import Counter

# Turn the images in the folders to numpy arrays for LLM Model training
training_read_file_path: str = r"C:/Users/Besitzer/Desktop/Python/AI Projects/Facial Emotion Recognition/train"
test_read_file_path: str = r"C:/Users/Besitzer/Desktop/Python/AI Projects/Facial Emotion Recognition/test"

def images_to_tensor(folder_path: str) -> list[torch.Tensor]:
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".jpg")]
    
    images: list = []

    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        img = Image.open(img_path)
        img_array = np.array(img)
        img_tensor = torch.from_numpy(img_array)
        images.append(img_tensor)

    return images

batch_size: int = 1024

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
test_angry_labels = torch.full((test_num_angry,), 0, dtype=torch.long)
test_disgust_labels = torch.full((test_num_disgust,), 1, dtype=torch.long)
test_fear_labels = torch.full((test_num_fear,), 2, dtype=torch.long)
test_happy_labels = torch.full((test_num_happy,), 3, dtype=torch.long)
test_neutral_labels = torch.full((test_num_neutral,), 4, dtype=torch.long)
test_sad_labels = torch.full((test_num_sad,), 5, dtype=torch.long)
test_surprise_labels = torch.full((test_num_surprise,), 6, dtype=torch.long)
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
training_disgust_images_stacked = images_to_tensor((training_read_file_path + "/disgust"))
training_fear_images = images_to_tensor((training_read_file_path + "/fear"))
training_happy_images = images_to_tensor((training_read_file_path + "/happy"))
training_neutral_images = images_to_tensor((training_read_file_path + "/neutral"))
training_sad_images = images_to_tensor((training_read_file_path + "/sad"))
training_surprise_images = images_to_tensor((training_read_file_path + "/surprise"))

training_angry_images_stacked = torch.stack(training_angry_images, dim=0)
training_disgust_images_stacked = torch.stack(training_disgust_images_stacked, dim=0)
training_fear_images_stacked = torch.stack(training_fear_images, dim=0)
training_happy_images_stacked = torch.stack(training_happy_images, dim=0)
training_neutral_images_stacked = torch.stack(training_neutral_images, dim=0)
training_sad_images_stacked = torch.stack(training_sad_images, dim=0)
training_surprise_images_stacked = torch.stack(training_surprise_images, dim=0)

"""
# Before combining them together in X_training, we should create augmented versions of them,
#   since there is too little disgust in the training set
def generate_augmented_versions(
        image_tensors: torch.Tensor, 
        num_versions: int
    ) -> torch.Tensor:

    # Apply slight randomizations on the images to create very similar, but not same datas.
    master_pipeline = T.Compose(
        [ 
            # Flip 10 % of the images horizontally.
            T.RandomHorizontalFlip(p=0.5),
            # Rotate the images by a random angle between -20 and +20 degrees.
            T.RandomRotation(degrees=10),
            # Randomly zooms in on the image between 80 % and 120 %
            T.RandomAffine(degrees=0, scale=(0.8, 1.2), shear=10),
            # Apply a random perspective(shearing) transformation for % 40 of the time.
            T.RandomPerspective(distortion_scale=0.2, p=0.4),
            # Randomly change the brightness and contrast of the image.
            T.ColorJitter(brightness=0.3, contrast=0.3),
            # Randomly erase a rectangular regions in the image.
            T.RandomErasing(p=0.4, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0),
        ]
    )
    
    all_augmented_images: list[torch.Tensor] = []
    num_original_images: int = image_tensors.shape[0]
    for _ in range(num_versions):
        # Randomly select one image to augment.
        random_ind: int = np.random.randint(0, num_original_images)
        single_image = image_tensors[random_ind]
        # Prepare the single image for transform pipeline.
        #   [48, 48] -> [1, 1, 48, 48]
        single_image_for_transform = single_image.unsqueeze(0).unsqueeze(0)
        # Apply the pipeline.
        augmented_single_image = master_pipeline(single_image_for_transform)
        # Squeeze the result back to original size and append to our list.
        augmented_single_image = augmented_single_image.squeeze()
        all_augmented_images.append(augmented_single_image)

    stacked_all_augmented_images = torch.stack(all_augmented_images, dim=0)

    return stacked_all_augmented_images

# 4000 Images will be choosen for training from each emotion.
augmented_training_disgust_images_stacked = generate_augmented_versions(training_disgust_images_stacked, 4000 - training_disgust_images_stacked.shape[0])

# Combine the newly created augmented images with the original ones.
training_disgust_images_stacked = torch.cat(
    [
        training_disgust_images_stacked,
        augmented_training_disgust_images_stacked
    ],
    dim=0
)

# Repeat the same for angry and surprise.
augmented_training_angry_images_stacked = generate_augmented_versions(training_angry_images_stacked, 4000 - training_angry_images_stacked.shape[0])
training_angry_images_stacked = torch.cat(
    [
        training_angry_images_stacked,
        augmented_training_angry_images_stacked
    ],
    dim=0
)
augmented_training_surprise_images_stacked = generate_augmented_versions(training_surprise_images_stacked, 4000 - training_surprise_images_stacked.shape[0])
training_surprise_images_stacked = torch.cat(
    [
        training_surprise_images_stacked,
        augmented_training_surprise_images_stacked
    ],
    dim=0
)
"""

X_training = torch.cat(
    [
        training_angry_images_stacked,
        training_disgust_images_stacked,
        training_fear_images_stacked,
        training_happy_images_stacked,
        training_neutral_images_stacked,
        training_sad_images_stacked,
        training_surprise_images_stacked,
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

training_angry_labels = torch.full((training_num_angry,), 0, dtype=torch.long)
training_disgust_labels = torch.full((training_num_disgust,), 1, dtype=torch.long)
training_fear_labels = torch.full((training_num_fear,), 2, dtype=torch.long)
training_happy_labels = torch.full((training_num_happy,), 3, dtype=torch.long)
training_neutral_labels = torch.full((training_num_neutral,), 4, dtype=torch.long)
training_sad_labels = torch.full((training_num_sad,), 5, dtype=torch.long)
training_surprise_labels = torch.full((training_num_surprise,), 6, dtype=torch.long)
all_training_labels: list = [
    training_angry_labels, training_disgust_labels, training_fear_labels,
    training_happy_labels, training_neutral_labels, training_sad_labels,
    training_surprise_labels,
]

y_training = torch.cat(all_training_labels, dim=0)

training_dataset = TensorDataset(X_training, y_training)

# We will create a custom sampler for training, since the dataset is unbalanced.

# Get the count of each emotion.
training_emotion_counts = torch.bincount(y_training)
# Calculate weight for each emotion.
training_emotion_weights: float = 1 / training_emotion_counts
# Assign a weight to every single sample in the training dataset.
weighted_labels = torch.tensor([training_emotion_weights[label] for label in y_training])

# Create the Sampler and DataLoaders.

# Create WeightedRandomSampler: It will draw samples with probabilities.
#   proportional to their weights.
sampler = WeightedRandomSampler(
    weights=weighted_labels,
    num_samples=len(weighted_labels), # Draw this many samples in total per epoch.
    replacement=True, # Allows oversampling of rare emotions.
)

training_loader = DataLoader(
    dataset = training_dataset,
    batch_size = batch_size,
    sampler = sampler,
)

print("\nAll data's are prepared and bundled into test- and training_loaders.")

if __name__ == "__main__":

    # See the imbalances in the dataset
    import matplotlib.pyplot as plt

    # --- This code assumes it's running after your data.py script ---
    # Or that the necessary variables (like the *_num_* variables) are available.

    # 1. Define the labels for clearer output
    emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

    # 2. Store the counts you've already calculated in a dictionary
    training_counts = {
        "Angry": training_num_angry,
        "Disgust": training_num_disgust,
        "Fear": training_num_fear,
        "Happy": training_num_happy,
        "Neutral": training_num_neutral,
        "Sad": training_num_sad,
        "Surprise": training_num_surprise
    }

    # 3. Calculate the total number of images
    total_training_images = sum(training_counts.values())

    # 4. Print the distribution details
    print("--- Training Set Distribution ---")
    print(f"Total images: {total_training_images}\n")

    for emotion, count in training_counts.items():
        percentage = (count / total_training_images) * 100
        print(f"{emotion:<10}: {count:>5} images ({percentage:.2f}%)")

    print("\n--- Visualization ---")
    # 5. Create and display a bar chart for visual inspection
    plt.figure(figsize=(10, 6))
    plt.bar(training_counts.keys(), training_counts.values(), color='skyblue')
    plt.title('Distribution of Emotions in Training Set')
    plt.xlabel('Emotion')
    plt.ylabel('Number of Images')
    # Add the count text on top of each bar
    for i, count in enumerate(training_counts.values()):
        plt.text(i, count + 50, str(count), ha='center') # Adjust the '50' for better positioning
    plt.show()
