import cv2 as cv
import sys

img = cv.imread('soccer.jpg')

if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY) # BGR 컬러 영상을 명암 영상으로 변환
gray_small=cv.resize(gray,dsize=(0,0),fx=0.5,fy=0.5) # 반으로 축소
# dsize는 변환할 크기이고 (0,0)으로 지정하면 비율을 지정하는 fx,fy에 따른다. 가로 세로를 반으로 만든다.

cv.imwrite('soccer_gray.jpg',gray) # soccer_gray이라는 파일을 생성하고 gray 객체를 저장한다.
cv.imwrite('soccer_gray_small.jpg',gray_small)

cv.imshow("Color image",img)
cv.imshow("Gray Image",gray)
cv.imshow("Gray image small",gray_small)

cv.waitKey()
cv.destroyAllWindows()