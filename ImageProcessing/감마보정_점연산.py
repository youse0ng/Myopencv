# 감마 보정

'''인간의 눈은 빛의 밝기 변화에 비선형적으로 반응 한다.
예를 들어, 명암 10을 20으로 올렸을 때와 120에서 130으로 올렸을 때 같은 양을 늘렸지만,
인간이 느끼는 밝아지는 정도는 두 경우가 다르다.

감마 보정은 이러한 비선형적인 시각 반응을 수학적으로 표현한 것이다.
감마 보정의 수식은 

f'(j,i) = (L-1) * F(j,i)**r

r은 감마 계수이고 사용자 정의
F(j,i)는 L-1로 Normalization한 화소값이다.

r>1이면 어둡게
r<1이면 밝게
r=1이면 영상을 유지
'''


import cv2 as cv
import numpy as np

img = cv.imread("ImageProcessing\soccer.jpg")
img = cv.resize(img,dsize=(0,0),fx=0.25,fy=0.25)

def gamma(f,gamma=1.0):
    f1=f/255.0 # L이 256이라고 가정 즉, 화소값을 0~ 256 까지만 갖는다는 의미
    return np.uint8(255*(f1**gamma))

gc = np.hstack((gamma(img,0.5),gamma(img,0.75),gamma(img,1.0),gamma(img,2.0),gamma(img,3.0)))

cv.imshow("gamma",gc)
cv.waitKey()
cv.destroyAllWindows()