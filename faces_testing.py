import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("./recognizers/face-trainner.yml")

labels = {"name": 1}
with open("pickles/face-labels.pickle", 'rb') as f:
    face_labels = pickle.load(f)
    labels = {v: k for k, v in face_labels.items()}

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.4, minNeighbors=4)
    for (x, y, w, h) in faces:
        gray_roi = gray[y:y+h, x:x+w]
        id_name, conf = recognizer.predict(gray_roi)
       
        if conf >= 15 and conf <= 95:
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_name]
            color = (204, 204, 0)
            stroke = 2
            cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)

        color = (204, 102, 0)
        stroke = 2
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, stroke)
        
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()