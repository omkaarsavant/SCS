from deepface import DeepFace
import cv2
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

# Load dataset of faces
dataset_path = 'E:\SCC\dataset'  # Update with your dataset path
faces = []
labels = []

for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)
    if os.path.isdir(person_path):
        for img_file in os.listdir(person_path):
            img_path = os.path.join(person_path, img_file)
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"Failed to load image: {img_path}")
                continue

            try:
                # Use DeepFace to get the face embeddings (disable face detection enforcement)
                embedding = DeepFace.represent(img, model_name="Facenet", enforce_detection=False)[0]["embedding"]
                faces.append(embedding)
                labels.append(person)
                print(f"Processed image: {img_path}")
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                continue

# Check if any embeddings were extracted
if len(faces) == 0:
    print("No embeddings were extracted. Check the dataset or image processing steps.")
    exit()

# Convert faces and labels to numpy arrays
faces = np.array(faces)
labels = np.array(labels)

# Encode labels (names)
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Train an SVM classifier on the entire dataset
classifier = SVC(kernel='linear', probability=True)
classifier.fit(faces, labels_encoded)

# Cross-validation to evaluate the model (optional, since the dataset is very small)
scores = cross_val_score(classifier, faces, labels_encoded, cv=5)  # 5-fold cross-validation
print(f'Cross-validation accuracy: {np.mean(scores) * 100:.2f}%')

# Save the trained classifier and label encoder
import joblib
joblib.dump(classifier, 'face_classifier.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
