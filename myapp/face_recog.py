import cv2
import numpy as np
import face_recognition
import os

#address = "http://192.168.1.3:8080/video"

path = 'D:\Face Recognition\my_env\Lib\Face_Recog\Images'
images = []
classNames = []
myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncoding(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncoding(images)
print('Encoding done')

cap = cv2.VideoCapture(0)
#cap.open(address)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    faceCurr = face_recognition.face_locations(imgS)
    encodeCurr = face_recognition.face_encodings(imgS, faceCurr)

    for encodeFace, faceLoc in zip(encodeCurr, faceCurr):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDist = face_recognition.face_distance(encodeListKnown, encodeFace)
        print(faceDist)
        matchIndex = np.argmin(faceDist)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLoc
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)


    cv2.imshow('Webcam',img)
    cv2.waitKey(1)




#faceLoc = face_recognition.face_locations(img)[0]
#encodeImg = face_recognition.face_encodings(img)[0]
#cv2.rectangle(img, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

#faceLocT = face_recognition.face_locations(imgTest)[0]
#encodeImgT = face_recognition.face_encodings(imgTest)[0]
#cv2.rectangle(imgTest, (faceLocT[3], faceLocT[0]), (faceLocT[1], faceLocT[2]), (255, 0, 255), 2)

#results = face_recognition.compare_faces([encodeImg], encodeImgT)
#faceDist = face_recognition.face_distance([encodeImg], encodeImgT)