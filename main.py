import cv2
import numpy as np

cap = cv2.VideoCapture(0)

human_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')

while True:
    _,img = cap.read()

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    humans = human_cascade.detectMultiScale(gray,1.9,1)

    for (x,y,w,h) in humans:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

    cv2.imshow("vid",img)

    if cv2.waitKey(1) & 0xFF==ord('q'):
        break