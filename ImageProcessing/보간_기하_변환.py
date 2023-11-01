import cv2 as cv

img = cv.imread('ImageProcessing\\rose.png')
patch = img[250:350,170:270,:]

img = cv.rectangle(img,pt1=(170,250),pt2=(270,350),color=(255,0,0),thickness=3)
patch1=cv.resize(patch,dsize=(0,0),fx=5,fy=5,interpolation=cv.INTER_NEAREST)
patch2=cv.resize(patch,dsize=(0,0),fx=5,fy=5,interpolation=cv.INTER_LINEAR)
patch3=cv.resize(patch,dsize=(0,0),fx=5,fy=5,interpolation=cv.INTER_CUBIC)

cv.imshow("Original",img)
cv.imshow('Resize nearest',patch1) # 최근접 이웃 보간
cv.imshow("Resize Linear",patch2) # 양선형 보간
cv.imshow("Resize bicubic",patch3) # 양3차 보간

cv.waitKey()
cv.destroyAllWindows()