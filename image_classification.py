import os
import cv2

# Paths
dataset_folder = r'C:\Users\Lenovo\Desktop\Dataset\Dataset'  # Folder containing the dataset
output_folder = r'C:\Users\Lenovo\Desktop\Face_Dataset'  # Folder to save labeled images

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Load images, label them, and save them in the output folder with labels in filenames
def label_and_save_images(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        img = cv2.imread(file_path)
        if img is not None:
            # Label face images (assuming face images start with "Human")
            if filename.lower().startswith('human'):
                new_filename = '1_' + filename  # Prefix with '1_' for face images
            else:  # Label non-face images
                new_filename = '0_' + filename  # Prefix with '0_' for non-face images

            # Ensure the extension (e.g., png) is preserved
            save_path = os.path.join(output_folder, new_filename)
            cv2.imwrite(save_path, img)  # Save the image with the new name

    print(f"Images labeled and saved to {output_folder}")

# Label the images and save them in the output folder
label_and_save_images(dataset_folder)
