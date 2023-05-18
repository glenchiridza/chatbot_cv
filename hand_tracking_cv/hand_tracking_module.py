import cv2
import mediapipe as mp
import time


class HandDetector:

    def __init__(self, s_image_mode=False,
                 max_hands=2, min_detect_conf=0.5,
                 min_track_conf=0.5):
        self.s_image_mode = s_image_mode
        self.max_hands = max_hands
        self.min_detect_conf = min_detect_conf
        self.min_track_conf = min_track_conf

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.s_image_mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.min_detect_conf,
            min_tracking_confidence=self.min_track_conf
        )

        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True, chosen_idx=0):
        self.lm_list = []
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[handNo]
            for idx, lm in enumerate(my_hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lm_list.append([idx, cx, cy])
                if draw and idx == chosen_idx:
                    cv2.circle(img, (cx, cy), 8, (255, 0, 255), cv2.FILLED)

        return self.lm_list

    thumb = 4
    index = 8
    middle_finger = 12
    ring_finger = 16
    pinky_finger = 20
    tip_ids = [thumb, index, middle_finger, ring_finger, pinky_finger]

    def fingersUp(self):
        fingers = []

        # checking for thumb on x-axis --1 according to our hand tracker lmlist
        if self.lm_list[self.tip_ids[0]][1] > self.lm_list[self.tip_ids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # for the other 4 fingers, uses y-axis -- 2
        for idx in range(1, 5):
            if self.lm_list[self.tip_ids[idx]][2] < self.lm_list[self.tip_ids[idx] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers


def main():
    pTime = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lm_list = detector.findPosition(img)

        if len(lm_list) != 0:
            print(lm_list[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (255, 0, 0), 2)

        cv2.imshow("image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
