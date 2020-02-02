import numpy as np
import cv2
import math
import datetime
cap=cv2.VideoCapture('traffic.mp4')
centerPositions=[]
global blobs
blobs=[]
INENTRY=0
OUTENTRY=0

kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(15,15))

ret,imgFrame1Copy=cap.read()
ret,imgFrame2Copy=cap.read()
carCount=0
twovehiclecount=0
blnFirstFrame = True

crossingLine=np.zeros((2,2),np.float32)
horizontalLinePosition=434
crossingLine[0][0]= 0
crossingLine[0][1]= horizontalLinePosition
crossingLine[1][0]= 1500
crossingLine[1][1] =horizontalLinePosition

class blobx(object): 
    def __init__(self,contour):
        global currentContour 
        global currentBoundingRect 
        global centerPosition
        global centerPositions
        global cx
        global cy
        global dblCurrentDiagonalSize 
        global dblCurrentAspectRatio 
        global intCurrentRectArea
        global blnCurrentMatchFoundOrNewBlob 
        global blnStillBeingTracked 
        global intNumOfConsecutiveFramesWithoutAMatch 
        global predictedNextPosition
        global numPositions
        self.predictedNextPosition=[]
        self.centerPosition=[]
        currentBoundingRect=[]
        currentContour=[]
        self.centerPositions=[]
        self.currentContour=contour
        self.currentBoundingArea=cv2.contourArea(contour)
        x,y,w,h = cv2.boundingRect(contour)
        self.currentBoundingRect=[x,y,w,h]
        cx=(2*x+w)/2
        cy=(2*y+h)/2
        self.centerPosition=[cx,cy]
        self.dblCurrentDiagonalSize=math.sqrt(w*w+h*h)
        self.dblCurrentAspectRatio=(w/(h*1.0))
        self.intCurrentRectArea=w*h
        self.blnStillBeingTracked = True
        self.blnCurrentMatchFoundOrNewBlob = True
        self.intNumOfConsecutiveFramesWithoutAMatch = 0
        self.centerPositions.append(self.centerPosition)    
    def predictNextPosition(self):

        numPositions=len(self.centerPositions)
        if (numPositions == 1):
            self.predictedNextPosition=[self.centerPositions[-1][-2],self.centerPositions[-1][-1]]
        if(numPositions >= 2):
            deltaX = self.centerPositions[1][0]-self.centerPositions[0][0]
            deltaY =self.centerPositions[1][1] -self.centerPositions[0][1]
            self.predictedNextPosition =[self.centerPositions[-1][-2]+deltaX,self.centerPositions[-1][-1]+deltaY ]
       
            self.predictedNextPosition=[self.centerPositions[-1][-2]+deltaX,self.centerPositions[-1][-1]+deltaY ]

def matchCurrentFrameBlobsToExistingBlobs(blobs,currentFrameBlobs):
    for existingBlob in blobs:
        existingBlob.blnCurrentMatchFoundOrNewBlob = False
        existingBlob.predictNextPosition()
    for currentFrameBlob in currentFrameBlobs:
        intIndexOfLeastDistance = 0
        dblLeastDistance = 1000000.0
        for i in range(len(blobs)):
            if (blobs[i].blnStillBeingTracked == True):
                dblDistance=distanceBetweenPoints(currentFrameBlob.centerPositions[-1],blobs[i].predictedNextPosition)
                if (dblDistance < dblLeastDistance):
                    dblLeastDistance = dblDistance
                    intIndexOfLeastDistance = i
        if (dblLeastDistance < (currentFrameBlob.dblCurrentDiagonalSize * 1.0)/1.2):
            blobs=addBlobToExistingBlobs(currentFrameBlob, blobs, intIndexOfLeastDistance)
        else:
            blobs,currentFrameBlob=addNewBlob(currentFrameBlob, blobs)
    for existingBlob in blobs:
        if (existingBlob.blnCurrentMatchFoundOrNewBlob == False):
            existingBlob.intNumOfConsecutiveFramesWithoutAMatch = existingBlob.intNumOfConsecutiveFramesWithoutAMatch + 1
        if (existingBlob.intNumOfConsecutiveFramesWithoutAMatch >=3):
            existingBlob.blnStillBeingTracked =False
    return blobs   


def distanceBetweenPoints(pos1,pos2):
    if (pos2==[]):
        dblDistance=math.sqrt((pos1[0])**2+(pos1[1])**2)
    else:
        dblDistance=math.sqrt((pos2[0]-pos1[0])**2+(pos2[1]-pos1[1])**2)
    return dblDistance
def addBlobToExistingBlobs(currentFrameBlob, blobs, intIndex):
    blobs[intIndex].currentContour = currentFrameBlob.currentContour
    blobs[intIndex].currentBoundingRect = currentFrameBlob.currentBoundingRect
    blobs[intIndex].centerPositions.append(currentFrameBlob.centerPositions[-1])
    blobs[intIndex].dblCurrentDiagonalSize = currentFrameBlob.dblCurrentDiagonalSize
    blobs[intIndex].blnStillBeingTracked = True
    blobs[intIndex].blnCurrentMatchFoundOrNewBlob = True
    return blobs
def addNewBlob(currentFrameBlob,Blobs):
    currentFrameBlob.blnCurrentMatchFoundOrNewBlob = True
    blobs.append(currentFrameBlob)
    return blobs,currentFrameBlob

def drawBlobInfoOnImage(blobs,f1):
    for i in range(len(blobs)):
        if (blobs[i].blnStillBeingTracked == True):
            x,y,w,h=blobs[i].currentBoundingRect
            cx=blobs[i].centerPositions[-1][-2]
            cy=blobs[i].centerPositions[-1][-1]
            cv2.rectangle(f1,(x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.circle(f1,(int(cx),int(cy)),2,(0,0,0),-1)
            text =str(i)
             
    return f1                             


def drawCarCountOnImage(carCount,twovehiclecount,f1):
    initText = "Car Counter: "
    text =initText+str(carCount) +  "   Bike Counter:"+str(     twovehiclecount)
    cv2.putText(f1, text, (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0 ,255), 2)
    cv2.putText(f1, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
        (10, f1.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
    return f1

def checkIfBlobsCrossedTheLine(blobs,horizontalLinePosition,carCount,twovehiclecount,INENTRY,OUTENTRY):
    atLeastOneBlobCrossedTheLine= False
    for blob in blobs:
        if (blob.blnStillBeingTracked == True and len(blob.centerPositions) >= 2):
            cx=blob.centerPositions[-1][-2]
            cy=blob.centerPositions[-1][-1]
            prevFrameIndex= len(blob.centerPositions) - 2
            currFrameIndex= len(blob.centerPositions) - 1
            if (blob.centerPositions[prevFrameIndex][-1] >= horizontalLinePosition and blob.centerPositions[currFrameIndex][-1] < horizontalLinePosition) and cx>231 and cx<1000:
                x,y,w,h=blob.currentBoundingRect
     
                if (w>100 and h>30):
                    carCount = carCount + 1
                    INENTRY=INENTRY+1
                else:
                    twovehiclecount=twovehiclecount+1
                    INENTRY=INENTRY+1
                atLeastOneBlobCrossedTheLine = True
            if (blob.centerPositions[prevFrameIndex][-1] <= horizontalLinePosition and blob.centerPositions[currFrameIndex][-1] > horizontalLinePosition) and cx>231 and cx<1000:
                x,y,w,h=blob.currentBoundingRect
                
                if (w>100 and h>30):
                    carCount = carCount + 1
                    OUTENTRY=OUTENTRY+1
                else:
                    twovehiclecount=twovehiclecount+1
                    OUTENTRY=OUTENTRY+1
                atLeastOneBlobCrossedTheLine = True
    return atLeastOneBlobCrossedTheLine,carCount,twovehiclecount,INENTRY,OUTENTRY

def masking(a):
    mask=np.zeros(a.shape,dtype=np.uint8)
    roi_corner=np.array([[(63,475),(443,488),(754,472),(1013,450),(843,417),(719,407),(605,399),(514,392),(412,393),(319,392),(265,428),(149,463)]],dtype=np.int32)
    channel_count=a.shape[2]
    ignore_mask_color=(255,)*channel_count
    cv2.fillPoly(mask,roi_corner,ignore_mask_color)
    masked_image=cv2.bitwise_and(a,mask)
    return masked_image

while (True):
   
    f1=imgFrame1Copy
    f2=imgFrame2Copy
    
    masked_image=masking(f1)
    s1=masked_image
    masked_image=masking(f2)
    s2=masked_image
    a1 = cv2.cvtColor(s1,cv2.COLOR_BGR2GRAY)
    b1 = cv2.cvtColor(s2,cv2.COLOR_BGR2GRAY)
    a2 = cv2.GaussianBlur(a1,(5,5),0)
    b2 = cv2.GaussianBlur(b1,(5,5),0)
    imgDifference=cv2.absdiff(b2,a2)
    ret1,th1 = cv2.threshold(imgDifference,30,255,cv2.THRESH_BINARY)
    th1 = cv2.dilate(th1,kernel,iterations = 1)
    th1 = cv2.dilate(th1,kernel,iterations = 1)
    fgmask = cv2.erode(th1,kernel,iterations = 1)
    frameNo=cap.get(1)
    fgmask = cv2.morphologyEx(th1, cv2.MORPH_OPEN, kernel1)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel2)
    fg2=np.zeros((fgmask.shape[0],fgmask.shape[1],3),np.uint8)
    fg3=np.zeros((fgmask.shape[0],fgmask.shape[1],3),np.uint8)
    _,contours, hierarchy = cv2.findContours(fgmask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(fg2, contours, -1, (255,255,255), -1)
    hulls=[]
    for c in range(len(contours)):
        hull=cv2.convexHull(contours[c])
        hulls.append(hull)
    curFrameblobs=[]
    for c in range(len(hulls)):
        ec=blobx(hulls[c])
        if(ec.intCurrentRectArea>100 and ec.dblCurrentDiagonalSize>30 and ec.currentBoundingRect[2]>20 and ec.currentBoundingRect[3]>20 and (ec.currentBoundingArea*1.0/ec.intCurrentRectArea)>.4):
            curFrameblobs.append(ec)
    contor=[]
    for af in curFrameblobs:
        contor.append(af.currentContour)
    if (blnFirstFrame ==True):
        for f1 in curFrameblobs:
            blobs.append(f1)
    else: 
        blobs=matchCurrentFrameBlobsToExistingBlobs(blobs,curFrameblobs)                     
    f1=drawBlobInfoOnImage(blobs,f1)
    atLeastOneBlobCrossedTheLine,carCount,twovehiclecount,INENTRY,OUTENTRY=checkIfBlobsCrossedTheLine(blobs, horizontalLinePosition, carCount,twovehiclecount,INENTRY,OUTENTRY)
    if (atLeastOneBlobCrossedTheLine):
        cv2.line(f1,(crossingLine[0][0],crossingLine[0][1]),(crossingLine[1][0],crossingLine[1][1]),(0,255,0), 2)
    else:
        cv2.line(f1,(crossingLine[0][0],crossingLine[0][1]),(crossingLine[1][0],crossingLine[1][1]),(0,0,255), 2)
    f1=drawCarCountOnImage(carCount,twovehiclecount,f1)
    cv2.imshow('original',f1)
    cv2.drawContours(fg3, contours, -1, (255,255,255), -1)
    cv2.drawContours(fg2, hulls, -1, (255,255,255), -1)
    prev=INENTRY
    prev1=OUTENTRY
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    imgFrame1Copy=imgFrame2Copy
    ret,imgFrame2Copy=cap.read()

    if not ret:
        break
    blnFirstFrame = False
    if cap.get(1)==6000:
        break
    if cap.get(1)==6002:
        break
        
cap.release()
cv2.destroyAllWindows()

