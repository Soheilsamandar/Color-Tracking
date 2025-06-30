import cv2 as cv 
import numpy as np 

def find_color_center(mask):
    contours,_ = cv.findContours(mask,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    center = None
    if contours:
        largest = max(contours,key=cv.contourArea)
        M = cv.moments(largest)
        if cv.contourArea(largest)>300:
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                center = (cx,cy)
    return center

def draw_line(cx,cy,height,width):
    color = (0,0,255)
    cv.line(frame,(cx,0),(cx,height),color,2)
    cv.line(frame,(0,cy),(width,cy),color,2)
    cv.circle(frame,(cx,cy),5,(0,0,0),-1)
    cv.circle(frame,(cx,cy),20,(0,255,0),1)
    cv.putText(frame,f"x:{cx},y:{cy}",(cx+10,cy-10),cv.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),2)

def draw_Contours(frame,mask):
    contours,_ = cv.findContours(mask,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv.contourArea(cnt)>300:
            rect = cv.minAreaRect(cnt)
            box = cv.boxPoints(rect)
            box = np.intp(box)
            cv.drawContours(frame,[box],0,(255,0,0),2)


cap = cv.VideoCapture(0)
while True:
    ret,frame = cap.read()
    hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)
    height,width = frame.shape[:2]
    default_cx = width//2
    default_cy = height//2

    lowR1 = np.array([0,120,70])
    upR1 = np.array([10,255,255])
    lowR2 = np.array([170,120,70])
    upR2 = np.array([180,255,255])
    mask_red = cv.inRange(hsv,lowR1,upR1)+cv.inRange(hsv,lowR2,upR2)

    lowG=np.array([40,50,50])
    upG = np.array([90,255,255])
    mask_green = cv.inRange(hsv,lowG,upG)
   
    lowB = np.array([100,150,50])
    upB = np.array([135,255,255])
    mask_B = cv.inRange(hsv,lowB,upB)
    draw_Contours(frame,mask_B)

    #center = find_color_center(mask_green) or find_color_center(mask_B)
    center = find_color_center(mask_B)
    if center:
        cx,cy = center
        draw_line(cx,cy,height,width)  
    else:
        cx,cy = default_cx,default_cy  
    
    cv.imshow("WEBCAM",frame)
    if cv.waitKey(1)==27:
        break
cv.destroyAllWindows()

