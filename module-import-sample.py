import cv2
import mediapipe as mp
import time
import HandTrackingmodule as htm
pTime = 0
cTime = 0
#Cam Capture
#Spawning camera creating obj etc
cap = cv2.VideoCapture(0)
detector = htm.handDetector()

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img)
    if len(lmList) !=0:
        print(lmList[4])

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    #Display FPS

    cv2.putText(img,str(int(fps)),(10,70), cv2.FONT_HERSHEY_PLAIN,3,(255,255,255),3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)