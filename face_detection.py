import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Paths
hog_features_file = r'C:\Users\Lenovo\Desktop\Processed Face Dataset\hog_features.pkl'

# HOG Parameters (same as training)
win_size = (64, 64)  # This should match the training image size
block_size = (16, 16)
block_stride = (8, 8)
cell_size = (8, 8)
nbins = 9

# Load HOG features and labels
with open(hog_features_file, 'rb') as f:
    features, labels = pickle.load(f)
print("HOG features loaded from file.")

# Check if the features and labels are not empty
if len(features) == 0 or len(labels) == 0:
    print("No features or labels available. Please ensure the dataset is correct and try again.")
    exit()

# Ensure there are at least two classes (1 and 0) in the labels
if len(np.unique(labels)) < 2:
    print("Error: The dataset contains only one class. Please ensure there are both face and non-face images.")
    exit()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
# X_train: The features used for training the model.
# X_test: The features used for testing the model.
# y_train: The corresponding labels for the training features.
# y_test: The corresponding labels for the testing features.
# Train the SVM model
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# Evaluate the model
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Load the pre-trained Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to extract HOG features from a given region of interest
def extract_hog_features(roi):
    # Resize ROI to match the training window size
    resized_roi = cv2.resize(roi, win_size)

    # Extract HOG features from the resized ROI
    hog_features = hog(resized_roi, orientations=nbins, pixels_per_cell=cell_size,
                       cells_per_block=(2, 2), block_norm='L2-Hys', transform_sqrt=True,
                       visualize=False, feature_vector=True)

    return hog_features

# Function to detect and track faces using Haar cascade and HOG + SVM
def detect_and_track_face(frame, svm_model):
    # Convert to grayscale (Haar cascade works better with grayscale images)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame using Haar cascade
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(64, 64))

    for (x, y, w, h) in faces:
        # Extract the region of interest (ROI)
        roi_gray = gray[y:y+h, x:x+w]

        # Extract HOG features from the ROI
        hog_features = extract_hog_features(roi_gray)

        # Reshape the features to match SVM input format
        hog_features = np.reshape(hog_features, (1, -1))

        # Predict using the trained SVM model
        prediction = svm_model.predict(hog_features)

        # If SVM predicts it as a face, draw the rectangle
        if prediction == 1:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Face Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame

# Open webcam feed and detect faces
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Detect faces in the frame
    processed_frame = detect_and_track_face(frame, svm)

    # Display the resulting frame
    cv2.imshow('Face Detection', processed_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
