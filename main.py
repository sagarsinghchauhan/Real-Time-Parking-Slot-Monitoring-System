import cv2
import pickle
import cvzone
import numpy as np

# video feed
cap = cv2.VideoCapture('carPark.mp4')

with open('CarParkPos','rb') as f:
    posList = pickle.load(f)

width ,height = 107,48

def checkParkingSpace(imgpro):

    spaceCounter = 0

    for pos in posList:
        x,y = pos


        imgCrop = imgpro[y:y+height,x:x+width]
        # cv2.imshow(str(x*y),imgCrop)
        # cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), (255, 0, 255), 2)
        count = cv2.countNonZero(imgCrop)
        # cvzone.putTextRect(img,str(count),(x,y+height-3),
        #                    scale=1,thickness=2,
        #                    offset = 0,
        #                    colorR = (0,0,255))


        if count <750:
            color = (0,255,0)
            tickness = 5
            spaceCounter +=1
        else:
            color = (0,0,255)
            tickness =2
        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), color,tickness)
        cvzone.putTextRect(img, str(count), (x, y + height - 3),
                                              scale=1,thickness=2,
                                              offset = 0,
                                              colorR = color)

    cvzone.putTextRect(img, f'Free: {spaceCounter}/{len(posList)}',
                       (50, 50),
                       scale=2, thickness=3,
                       offset=10,
                       colorR = (0,255,0)
                       )


while True:
    if cap.get(cv2.CAP_PROP_POS_FRAMES)  == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    ret,img = cap.read()
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray,(3,3),1)
    # cv2.imshow("Image",img_blur)
    imgThreshold = cv2.adaptiveThreshold(img_blur,250,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,25,16)
    # cv2.imshow("Image_thres",imgThreshold)

    imgMedian = cv2.medianBlur(imgThreshold,5)
    # cv2.imshow("ImageMedian",imgMedian)

    kernal = np.ones((4,4),np.uint8)
    imgDilate = cv2.dilate(imgMedian, kernal, iterations=2)
    cv2.imshow("ImageDilate",imgDilate)




    checkParkingSpace(imgMedian)

    # for pos in posList:
    #     cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), (255, 0, 255), 2)







    cv2.imshow('frame',img)

    if cv2.waitKey(10) & 0xFF == 27:  # ESC to exit
        break