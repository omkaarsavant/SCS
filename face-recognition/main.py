import pickle
import cv2
import face_recognition
import numpy as np
import cvzone

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# backgroundImg = cv2.imread('Resources/background.jpg')

# Load the encoding file
print('Loading Encoded File')
with open('encodefile.p', 'rb') as file:
    encodeListKnownwithIds = pickle.load(file)
print('Encode File Loaded')

encodeListKnown, studentIds = encodeListKnownwithIds

while True:
    success, image = cap.read()
    if not success:
        print("Failed to grab frame")
        break
    backgroundImg = image

    imgS = cv2.resize(image, None, fx=0.25, fy=0.25)
    imageRGB = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imageRGB)
    encodeCurFrame = face_recognition.face_encodings(imageRGB, faceCurFrame)

    if not faceCurFrame:  # Check if no faces were detected
        print("No face detected")
    else:
        for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDist = face_recognition.face_distance(encodeListKnown, encodeFace)

            matchIndex = np.argmin(faceDist)

            if matches[matchIndex]:
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

                # Ensure bbox coordinates are within image dimensions
                if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
                    continue

                bbox = x1, y1, x2 - x1, y2 - y1

                backgroundImg = cvzone.cornerRect(backgroundImg, bbox, rt=0)

                # Display student ID above the bounding box
                cv2.putText(backgroundImg, str(studentIds[matchIndex]), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 2)

                print(f'Known Face Detected: {studentIds[matchIndex]}')
            else:
                print('Unknown Face Detected')

    cv2.imshow('Face', backgroundImg)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
