from flask import Flask, render_template, Response
import cv2
import numpy as np
import os
from deepface import DeepFace
from scipy.spatial.distance import cosine  # Import cosine similarity function

app = Flask(__name__)

# Load the DNN model for face detection
modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
configFile = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# Open the webcam
cap = cv2.VideoCapture(0)

# Load the images from dataset folder
dataset_path = "dataset"
folders = ['101', '102']  # Folder names for labeling
faces_db = {}

# Iterate over folders and load the images for comparison
for folder in folders:
    folder_path = os.path.join(dataset_path, folder)
    faces_db[folder] = []
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        try:
            # DeepFace will be used to compare the embeddings of these images later
            face_representation = DeepFace.represent(img_path, model_name="VGG-Face", enforce_detection=False)[0]["embedding"]
            faces_db[folder].append(face_representation)
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")

# Function to compare a face with the dataset
def recognize_face(face_img):
    try:
        face_rep = DeepFace.represent(face_img, model_name="VGG-Face", enforce_detection=False)[0]["embedding"]
        
        # Compare with each face in the dataset
        for label, representations in faces_db.items():
            for stored_rep in representations:
                # Use cosine similarity to compare embeddings
                similarity = 1 - cosine(stored_rep, face_rep)
                if similarity > 0.6:  # Adjust similarity threshold if needed
                    return label
    except Exception as e:
        print(f"Error recognizing face: {e}")
    return 'unknown'

def generate_frames():
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Get frame dimensions
        h, w = frame.shape[:2]

        # Prepare the image for the neural network
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

        # Pass the blob through the network
        net.setInput(blob)
        detections = net.forward()

        # Loop through the detections and process each face
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  # Confidence threshold
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")

                # Crop the face from the frame
                face_img = frame[y:y1, x:x1]

                # Recognize the face
                label = recognize_face(face_img)

                # Draw a rectangle around the face and display the label
                color = (0, 255, 0) if label != 'unknown' else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x1, y1), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    # Rendering an HTML template for displaying the video feed
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # Streaming the video feed to the browser
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
