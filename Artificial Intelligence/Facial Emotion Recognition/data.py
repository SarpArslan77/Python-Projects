
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
import torch
import torchvision.transforms as T
from torchvision.datasets import ImageFolder

# Custom Dataset for utilizing data augmentation on specific classes
class ConditionalAugmentationDataset(ImageFolder):
    def __init__(self, root, special_class_indexes, special_transform=None, standard_transform=None):
        super().__init__(root, transform=None)
        # transform = None, because we will manually apply it.

        self.special_class_indexes = special_class_indexes
        self.special_transform = special_transform
        self.standard_transform = standard_transform

    def __getitem__(self, index):
        # Get the path and label for the image at the given index.
        path, target = self.samples[index]
        # Load the image from the path.
        sample = self.loader(path)

        # Check if the image's label matches the special class index.
        if target in self.special_class_indexes:
            # If yes, apply augmentation.
            if self.special_transform:
                sample = self.special_transform(sample)
        else:
            # For all the other classes, apply the standard transform.
            if self.standard_transform:
                sample = self.standard_transform(sample)
        
        return sample, target


# Paths to training and test files
TRAINING_READ_FILE_PATH: str = r"C:/Users/Besitzer/Desktop/Python/AI Projects/Facial Emotion Recognition/train"
TEST_READ_FILE_PATH: str = r"C:/Users/Besitzer/Desktop/Python/AI Projects/Facial Emotion Recognition/test"

BATCH_SIZE: int = 128

# Apply slight randomizations on the images to create very similar, but not same datas.
augmentation_pipeline = T.Compose(
    [ 
        # Convert all images to 1-channel grayscale.
        T.Grayscale(num_output_channels=1),
        # Flip 90 % of the images horizontally.
        T.RandomHorizontalFlip(p=0.8),
        # Rotate the images by a random angle between -15 and +15 degrees.
        T.RandomRotation(degrees=15),
        # Randomly zooms in on the image between 80 % and 120 %
        #? T.RandomAffine(degrees=0, scale=(0.8, 1.2), shear=10),
        # Apply a random perspective(shearing) transformation for % 40 of the time.
        #? T.RandomPerspective(distortion_scale=0.2, p=0.4),
        # Randomly change the brightness and contrast of the image.
        T.ColorJitter(brightness=0.5, contrast=0.5),
        # Determine the size to 48 x 48.
        T.Resize(size=(48, 48)),
        # Always end with converting to tensor and normalizing.
        T.ToTensor(), # Converts to [0, 1] range and shape [C, H, W]
        # Randomly erase a rectangular regions in the image.
        #? T.RandomErasing(p=0.4, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0),
        #       RandomErasing requires a Tensor to work on.
        T.Normalize(mean=(0.5,), std=(0.5,)), # Normalizes to [-1, 1] range
    ]
)

# Define a seperate pipeline for testing.
standard_pipeline = T.Compose(
    [
        T.Grayscale(num_output_channels=1),
        T.Resize((48, 48)),
        T.ToTensor(),
        T.Normalize(mean=(0.5,), std=(0.5,),),
    ]
)

# Load the Dataset using ImageFolder with pipeline
try:
    train_dataset = ConditionalAugmentationDataset(
        root = TRAINING_READ_FILE_PATH,
        special_class_indexes = [1],
        special_transform = augmentation_pipeline,
        standard_transform = standard_pipeline,
    )
except:
    print("Training dataset couldn't be loaded!")
    exit()

# bincount function can only search in tensors.
targets = torch.tensor(train_dataset.targets)

# We will create a custom sampler for training, since the dataset is unbalanced.

# Get the count of each emotion.
training_emotion_counts: torch.Tensor = torch.bincount(targets)
# Calculate weight for each emotion.
try:
    training_emotion_weights: torch.Tensor = 1.0 / training_emotion_counts
except ZeroDivisionError:
    print(f"Error: training_emotion_counts is zero, cannot divide by zero!")
    exit()
# Assign a weight to every single sample in the training dataset.
weighted_labels = training_emotion_weights[targets]

# Create WeightedRandomSampler: It will draw samples with probabilities proportional to their weights.
sampler = WeightedRandomSampler(
    weights = weighted_labels,
    num_samples = len(weighted_labels), # Draw this many samples in total per epoch.
    replacement = True, # Allows oversampling of rare emotions.
)

# Create DataLoader with the Dataset and Sampler
train_loader = DataLoader(
    dataset = train_dataset,
    batch_size = BATCH_SIZE,
    #! sampler = sampler, # The sampler handles the shuffling logic.
    shuffle = True,
)

# Repeat the creating dataset for test files.
full_test_dataset = ImageFolder(
    root = TEST_READ_FILE_PATH,
    transform = standard_pipeline
)

# We will create a validation set for dynamically adjusting learning rates while training.

# Define split sizes for test and validation.
test_subset_size: int = int(0.8 * len(full_test_dataset))
val_size: int = len(full_test_dataset) - test_subset_size

# Split the full test dataset into test and validation sets.
test_subset, val_subset = random_split(full_test_dataset, [test_subset_size, val_size])

test_loader = DataLoader(
    dataset = test_subset,
    batch_size = BATCH_SIZE,
    shuffle = False,
)
val_loader = DataLoader(
    dataset = val_subset,
    batch_size = BATCH_SIZE,
    shuffle = False,
)

print("\nData pre-processing was successful.")


