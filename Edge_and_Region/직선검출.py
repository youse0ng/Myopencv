import cv2 as cv
import numpy as np

img = cv.imread("Edge_and_Region\soccer.jpg")
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
canny = cv.Canny(gray,70,170) # 캐니 알고리즘 (NMS:비최대 억제) # 미분을 통한 선검출

contour,hierarchy = cv.findContours(canny,cv.RETR_LIST,cv.CHAIN_APPROX_NONE)
print(len(contour))
lcontour=[]
for i in range(len(contour)): # 길이가 100보다 크면
    if contour[i].shape[0] > 100:
        lcontour.append(contour[i])

cv.drawContours(img,lcontour,-1,(0,255,0),2)
cv.imshow("Original with contours",img)
cv.imshow('Canny',canny)

cv.waitKey()
cv.destroyAllWindows()

'''이웃한 엣지를 연결하여 경계선으로 검출. 이런 경우 자잘하게 끊기는 일이 자주 발생되는 문제가 생김
그래서 나온 허프 변환
'''