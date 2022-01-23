import cv2
import mediapipe as mp
import time

#Spawning camera creating obj etc
cap = cv2.VideoCapture(0)

#####Kynan test Sphere
#Change this to add a circle to the id yu want to check
idCheck = 0

#The use of this is, we will create a list which will point to an ID but also the position, so you could check where the finger tip is etc.
######################
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils


pTime = 0
cTime = 0
#Cam Capture
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #verify
    print(results.multi_hand_landmarks)

    #id = 'Bone id etc '

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                #print(id,lm)
                #Getting Height Width and Channel (Decimal to Pixel Conversion)
                h, w, c= img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                #printing positions per ID
                print(id, cx, cy)

                #if id == 0:

                    #Getting the position info
                    #cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    #FPS Draw
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    #Display FPS

    cv2.putText(img,str(int(fps)),(10,70), cv2.FONT_HERSHEY_PLAIN,3,(255,255,255),3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

    
