import cv2 as cv
import sys

img=cv.imread("Chapter2\girl_laughing.jpg")

if img is None:
    sys.exit("파일이 존재하지 않습니다.")

cv.rectangle(img,(830,30),(1000,200),(0,0,255),2) # 직사각형 그리기

cv.putText(img,'laugh',(830,24),cv.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2) # 글씨 쓰기
cv.imshow('Draw',img)

cv.waitKey()
cv.destroyAllWindows()