import cv2
import mediapipe as mp
import time

video = cv2.VideoCapture(0)
pTime = 0
while True:
    success,img = video.read()

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 0, 0), 3)

    cv2.imshow("glen_cam", img)
    cv2.waitKey(1)

