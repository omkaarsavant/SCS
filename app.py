from flask import Flask, render_template, Response
import cv2
import numpy as np

app = Flask(__name__)

# Load the DNN model for face detection
modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
configFile = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# Open the webcam
cap = cv2.VideoCapture(0)

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

        # Loop through the detections and draw rectangles around detected faces
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  # Confidence threshold
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")

                # Draw a rectangle around the face
                cv2.rectangle(frame, (x, y), (x1, y1), (255, 0, 0), 2)

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
