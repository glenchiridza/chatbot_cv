import cv2
import mediapipe as mp
import time
import os

import hand_tracking_cv.hand_tracking_module as htm

wCam, hCam = 640, 480
video = cv2.VideoCapture(0)
video.set(3, wCam)
video.set(4, hCam)
pTime = 0

detector = htm.HandDetector(min_detect_conf=0.75)

tips_ids = [4, 8, 12, 16, 20]

while True:
    success, img = video.read()

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 0, 0), 3)

    cv2.imshow("glen_cam", img)
    cv2.waitKey(1)
