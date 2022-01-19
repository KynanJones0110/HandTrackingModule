import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self,mode=False,maxHands = 2, detectionCon=0.5,trackCon=0.5 ):
        self.mode = mode
        self.maxhands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        #####Kynan test Sphere
        #Change this to add a circle to the id yu want to check
        idCheck = 0

        #The use of this is, we will create a list which will point to an ID but also the position, so you could check where the finger tip is etc.
        ######################
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.hands(self.mode,self.maxhands,self.detectionCon,self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils


    def findhands(self, img, draw =True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        #verify
        print(self.results.multi_hand_landmarks)

        #id = 'Bone id etc '

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
         
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                #print(id,lm)
                #Getting Height Width and Channel (Decimal to Pixel Conversion)
                h, w, c= img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                #printing positions per ID
                #print(id, cx, cy)
                lmList.append([id, cx, cy])
                    #Getting the position info
                if draw:
                    cv2.cir(img, (cx,cy), 15, (255,0,255),cv2.FILLED)

        return lmList
        


def main():
    pTime = 0
    cTime = 0
    #Cam Capture
    #Spawning camera creating obj etc
    cap = cv2.VideoCapture(0)
    detector = handDetector()

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

if __name__ == "__main__":
    main()
