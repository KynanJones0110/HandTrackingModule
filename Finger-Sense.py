import cv2
import time
import os


wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

#Image Path Dir

folderPath = "FingerImages"
myList = os.listdir()
print(myList)
overlayList = []

for imPath in myList:
    image = cv2.imread(f"{folderPath}/{imPath}1.jpg")
    overlayList.append(image)


#print (len(overlayList))



while True:

    success, img = cap.read()
    #Image size etc 
    img[0:200,0:200] = overlayList[0]

    cv2.imshow("Image", img)
    cv2.waitKey(1)
