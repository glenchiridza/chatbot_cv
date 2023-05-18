import cv2
import mediapipe as mp
import time
import os

import hand_tracking_cv.hand_tracking_module as htm

# import chat bot functions
from runner import response_msg

wCam, hCam = 640, 480
video = cv2.VideoCapture(0)
video.set(3, wCam)
video.set(4, hCam)
pTime = 0

detector = htm.HandDetector(min_detect_conf=0.75)

tips_ids = [4, 8, 12, 16, 20]

while True:
    success, img = video.read()
    img = detector.findHands(img)
    lm_list = detector.findPosition(img, draw=False)

    if len(lm_list) != 0:
        fingers = []

        # for thumb finger
        if lm_list[tips_ids[0]][1] > lm_list[tips_ids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # for the other 4 fingers
        for idx in range(1, 5):
            if lm_list[tips_ids[idx]][2] < lm_list[tips_ids[idx] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        total_fingers = fingers.count(1)
        print(total_fingers)

        if total_fingers == 0:

            cv2.putText(img, "well we have got nothing for you", (10, 70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,
                        (255, 0, 0), 2)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 0, 0), 3)

    cv2.imshow("glen_bot_cam", img)
    cv2.waitKey(1)
