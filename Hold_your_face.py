# Python code for Multiple Color Detection


import numpy as np
import cv2
print(cv2.__file__)

class Solution:
    def isRectangleOverlap(self, Rec1, Rec2):
        if (Rec1[0]>=Rec2[2]) or (Rec1[2]<=Rec2[0]) or (Rec1[3]<=Rec2[1]) or (Rec1[1]>=Rec2[3]):
            return False
        else:
            return True


class Rectangle:
    def __init__(self, min_x, max_x, min_y, max_y):
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y

    def is_intersect(self, other):
        if self.min_x > other.max_x or self.max_x < other.min_x:
            return False
        if self.min_y > other.max_y or self.max_y < other.min_y:
            return False
        return True

# Capturing video through webcam
webcam = cv2.VideoCapture(0)
r1 = [0,0,0,0]
r2 = [0,0,0,0]
# Start a while loop
em_img = cv2.imread("emoji.jpg", cv2.IMREAD_COLOR)
det_fac = cv2.imread("emoji1.jpg", cv2.IMREAD_COLOR)

face_cascade = cv2.CascadeClassifier('C:\\Users\\Nabeel\\AppData\\Local\\Programs\\Python\\Python36\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_alt2.xml')
while(1):
	
    # Reading the video from the
    # webcam in image frames
    _, imageFrame = webcam.read()
    gray = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    #for (x, y, w, h) in faces:
        #cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
    if(len(faces) != 0):
        x, y, w, h = faces[0]
        det_fac = imageFrame[y:y + h, x:x + w]
        imageFrame[y:y + h , x:x + w] = det_fac
        #em_fac = cv2.resize(em_img, (h, w))
        #imageFrame[y:y + h,x:x + w] = em_fac

        det_fac = cv2.resize(det_fac, (h, w))
       # imageFrame[y:y + h , x:x + w] = det_fac
        

    

    # Convert the imageFrame in
    # BGR(RGB color space) to
    # HSV(hue-saturation-value)
    # color space
    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

    # Set range for red color and
    # define mask
    red_lower = np.array([136, 87, 111], np.uint8)
    red_upper = np.array([180, 255, 255], np.uint8)
    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)

    # Set range for green color and
    # define mask
    green_lower = np.array([25, 52, 72], np.uint8)
    green_upper = np.array([102, 255, 255], np.uint8)
    green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)

    # Set range for blue color and
    # define mask
    blue_lower = np.array([94, 80, 2], np.uint8)
    blue_upper = np.array([120, 255, 255], np.uint8)
    blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)



    yellow_lower = np.array([20, 100, 100], np.uint8)
    yellow_upper = np.array([30, 255, 255], np.uint8)
    yellow_mask = cv2.inRange(hsvFrame, yellow_lower, yellow_upper)
    # Morphological Transform, Dilation
    # for each color and bitwise_and operator
    # between imageFrame and mask determines
    # to detect only that particular color
    kernal = np.ones((5, 5), "uint8")
    
    # For red color
    red_mask = cv2.dilate(red_mask, kernal)
    res_red = cv2.bitwise_and(imageFrame, imageFrame,
                            mask = red_mask)
    
    # For green color
    green_mask = cv2.dilate(green_mask, kernal)
    res_green = cv2.bitwise_and(imageFrame, imageFrame,mask = green_mask)
    
    # For blue color
    yellow_mask = cv2.dilate(yellow_mask, kernal)
    res_blue = cv2.bitwise_and(imageFrame, imageFrame,mask = blue_mask)

    # Creating contour to track red color
    contours, hierarchy = cv2.findContours(red_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            #imageFrame = cv2.rectangle(imageFrame, (x, y),(x + w, y + h),(0, 0, 255), 2)
            r1 = [x,y,x+w,y+h]
            det_fac = cv2.resize(det_fac, (w, h))
            imageFrame[ y:y + h , x:x + w] = det_fac
            #cv2.putText(imageFrame, "Pink Colour", (x, y),cv2.FONT_HERSHEY_SIMPLEX, 1.0,(0, 0, 255)) 

    # Creating contour to track green color
##    contours, hierarchy = cv2.findContours(green_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
##    
##    for pic, contour in enumerate(contours):
##        area = cv2.contourArea(contour)
##        if(area > 300):
##            x, y, w, h = cv2.boundingRect(contour)
##            imageFrame = cv2.rectangle(imageFrame, (x, y),
##                                    (x + w, y + h),
##                                    (0, 255, 0), 2)
##            
##            cv2.putText(imageFrame, "Green Colour", (x, y),
##                        cv2.FONT_HERSHEY_SIMPLEX,
##                        1.0, (0, 255, 0))

    # Creating contour to track blue color
    contours, hierarchy = cv2.findContours(yellow_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            #imageFrame = cv2.rectangle(imageFrame, (x, y),(x + w, y + h),(0 , 255, 255), 2)
            #cv2.putText(imageFrame, "Yellow Colour", (x, y),cv2.FONT_HERSHEY_SIMPLEX,1.0, (0, 255, 255))
            r2 = [x,y,x+w,y+h]
            ob = Solution()
            #if(ob.isRectangleOverlap(r1,r2) == True):
                #print("Winner Winner Chicken Dinner")

                
                    
            
    # Program Termination
    cv2.imshow("Carry your Face Game by Nabeel", imageFrame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
            webcam.release()
            cv2.destroyAllWindows()
            break

