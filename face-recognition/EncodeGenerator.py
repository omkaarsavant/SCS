from base64 import encode

import cv2
import face_recognition
import pickle
import os

print("Helllo")
imgFolderPath = 'Images'
PathList = os.listdir(imgFolderPath)
imgList = []
studentIds = []
for path in PathList:
    imgPath = os.path.join(imgFolderPath, path)
    imgList.append(cv2.imread(imgPath))
    studentIds.append(os.path.splitext(path)[0])
print(studentIds)


def findEncodings(imagesList):
    encodedList = []
    for img in imagesList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodedList.append(encode)
    return  encodedList

print('Encoding Started ...')
encodeListKnown = findEncodings(imgList)
encodeListKnownwithIds = [encodeListKnown, studentIds]
print('Encoding Finished ...')

encodeFile = open('encodefile.p', 'wb')
pickle.dump(encodeListKnownwithIds, encodeFile)
encodeFile.close()
print('File Saved ...')