# OpenCV를 잘 다루려면 numpy에 익숙해야한다.
import numpy as np
import cv2 as cv
import sys

img = cv.imread('ImageProcessing\soccer.jpg')

if img is None: # 이미지가 없다면
    sys.exit("이미지 파일이 없어요")
print(img.shape)
cv.imshow("original_RGB",img) # 3차원
cv.imshow("Upper left half",img[0:img.shape[0]//2,0:img.shape[1]//2,:]) # height,width,channel
cv.imshow("Center half",img[img.shape[0]//4:3*img.shape[0]//4,img.shape[1]//4:3*img.shape[1]//4,:])
# img[237:711,358:1075,:]
cv.imshow("Customized center Half",img[233:711,358:1075,:])
cv.imshow("R Channel",img[:,:,2])
cv.imshow("G Channel",img[:,:,1])
cv.imshow("B Channel",img[:,:,0])
# Opencv는 BGR, RGB가 아니다
cv.waitKey()
cv.destroyAllWindows()