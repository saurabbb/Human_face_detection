import os
import cv2
import numpy as np
from skimage.feature import hog
import pickle

# Paths
dataset_path = r'C:\Users\Lenovo\Desktop\Face_Dataset'
processed_path = r'C:\Users\Lenovo\Desktop\Processed Face Dataset'
hog_features_file = os.path.join(processed_path, 'hog_features.pkl')

# HOG Parameters
cell_size = (8, 8)  # Size of cells
block_size = (2, 2)  # Reduced block size to reduce the feature vector length
nbins = 9  # Number of orientation bins
resize_to = (64, 64)  # Resize images to this resolution

# Load Images and Labels with Resizing
def load_images_from_folder(folder):
    images = []
    labels = []
    if not os.path.exists(folder):
        print(f"Folder does not exist: {folder}")
        return images, labels

    file_count = 0  # Count how many files are being processed
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path):
            img = cv2.imread(file_path)
            if img is not None:
                # Resize the image to reduce memory usage
                img_resized = cv2.resize(img, resize_to)

                # Labeling based on filename:
                if filename.startswith('1_'):
                    label = 1  # Face
                elif filename.startswith('0_'):
                    label = 0  # Non-face
                else:
                    print(f"Skipping file: {filename} (Unknown label)")
                    continue

                images.append(img_resized)
                labels.append(label)
                file_count += 1
            else:
                print(f"Failed to load image: {filename}")
        else:
            print(f"File not found or not an image file: {file_path}")

    print(f"Total files processed: {file_count}")
    return images, labels

# HOG Feature Extraction
def extract_hog_features(images):
    hog_features = []
    for idx, img in enumerate(images):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h = hog(gray, orientations=nbins, pixels_per_cell=cell_size,
                cells_per_block=block_size, block_norm='L2-Hys',
                transform_sqrt=True, visualize=False, feature_vector=True)
        hog_features.append(h)
        print(f"Extracted HOG features from image {idx + 1}/{len(images)}")
    return np.array(hog_features)

# Create processed path if it doesn't exist
if not os.path.exists(processed_path):
    os.makedirs(processed_path)

# Load and preprocess the dataset
print("Loading dataset...")
images, labels = load_images_from_folder(dataset_path)

if len(images) == 0:
    print("No images found in the dataset. Please check the dataset path.")
else:
    # Extract HOG features
    print("Extracting HOG features...")
    features = extract_hog_features(images)
    labels = np.array(labels)

    # Save the extracted features and labels
    with open(hog_features_file, 'wb') as f:
        pickle.dump((features, labels), f)
    print(f"HOG features and labels saved to {hog_features_file}")

