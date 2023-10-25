'''가우시안 스무딩과 엠보싱하기
가우시안 블러를 통해 스무딩은 잡음을 제거하는 역할을 하지만, 물체의 경계를 흐릿하게 하는 블러링 부작용이 있다.
엠보싱은 물체에 돋을새김 느낌을 주는 엠보싱 필터이다.
'''

import cv2 as cv
import numpy as np

img = cv.imread("ImageProcessing\soccer.jpg")
img = cv.resize(img,dsize=(0,0),fx=0.4,fy=0.4)
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.putText(gray,'soccer',(10,20),cv.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
cv.imshow('Original',gray)

smooth = np.hstack((cv.GaussianBlur(gray,(5,5),0.0),cv.GaussianBlur(gray,(9,9),0.0),cv.GaussianBlur(gray,(15,15),0.0)))
# 첫번째 인수는 스무딩을 적용할 영상이고, 두번째 인수는 필터의 크기, 세번째 인수는 표준 편차인데, 0.0으로 설정하면 필터 크기를 보고 자동으로 추정한다.
cv.imshow("Smooth",smooth)

filter_emboss=np.array([[-1.0,-1.0,0.0],
                  [-1.0,0.0,1.0],
                  [0.0,1.0,1.0]])
# 엠보싱 필터에 대해 정의한다.

gray16 = np.int16(gray)
emboss = np.uint8(np.clip(cv.filter2D(gray16,-1,filter_emboss)+128,0,255)) # 두 번째 인수를 -1 로 하면 output image의 데이터형을 input_image와 동일하게 가져감
emboss_bad=np.uint8(cv.filter2D(gray16,-1,filter_emboss)+128) # -1 will give the output image depth as same as the input image 여기서 깊이는 channel을 얘기하는 걸까?
emboss_worse=cv.filter2D(gray,-1,filter_emboss) # 책에서는 주어진 영상 배열과 같은 형의 배열을 출력한다 했으니까 음.. 같은 데이터형을 출력한다고 생각해야겠다.
# 첫번째 인수는 source image 적용할 이미지, 세번째 인수는 kernel을 입력하는데 우리는 이미 사용자 지정한 filter(kernel)를 넣어준다.
# filter2D의 결과값은 gray16과 동일한 데이터형으로 나올것이다..
print(cv.filter2D(gray16,-1,filter_emboss).dtype) # 역시 gray16과 같은 int16을 출력한다.
cv.imshow("Emboss",emboss)
cv.imshow("Emboss_bad",emboss_bad)
cv.imshow("Emboss_worse",emboss_worse)

cv.waitKey()
cv.destroyAllWindows()