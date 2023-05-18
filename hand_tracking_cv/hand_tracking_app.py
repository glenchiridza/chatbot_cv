import hand_tracking_cv.hand_tracking_module as htm
import time
import cv2


def main():
    pTime = 0

    cap = cv2.VideoCapture(0)
    detector = htm.HandDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lm_list = detector.findPosition(img=img, draw=True,chosen_idx=4)

        if len(lm_list) != 0:
            print(lm_list[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (255, 0, 255), 2)

        cv2.imshow("image", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
