import cv2 as cv
import numpy as np
'''CV_8U: 이미지 픽셀값을 uint8로 설정
   CV_16U: 이미지 픽셀값을 uint16로 설정
   CV_32F: 이미지 픽셀값을 float32로 설정
   CV_64F: 이미지 픽셀값을 float64로 설정
'''
img=np.array([[0,0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,0],
              [0,0,0,1,0,0,0,0,0,0],
              [0,0,0,1,1,0,0,0,0,0],
              [0,0,0,1,1,1,0,0,0,0],
              [0,0,0,1,1,1,1,0,0,0],
              [0,0,0,1,1,1,1,1,0,0],
              [0,0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0,0,0]],dtype=np.float32)

ux = np.array([[-1,0,1]]) # x에 대한 미분 필터
uy = np.array([[-1,0,1]]).transpose() # y에 대한 미분 필터
k = cv.getGaussianKernel(3,1) # 첫번째 인수는 3 * 1 shape의 가우시안 필터 커널이고, 두 번째 인수는 sigma (표준 편차) = 1 이다.
gausianfilter = np.outer(k,k.transpose()) # 내가 알던 선형 대수학에서 배웠던 외적과는 조금 다른 결과값이 나오는 것 같다. 13i+34j+12k 이렇게 계산돼서 나왔는데,
# np.outer함수는 3차원 배열의 외적결과가 나온다. # np.outer의 외적 함수를 계산하는 과정을 인지할 필요가 있겠다.

dy = cv.filter2D(img,cv.CV_32F,uy) # img를 uy로 1차 미분한 값 (y 방향 미분: 수직 방향 미분) -> 수직의 픽셀값의 변화가 있는 곳을 볼 수 있음
dx = cv.filter2D(img,cv.CV_32F,ux) # img를 ux로 1차 미분한 값 (x 방향 미분: 수평 방향 미분) -> 수평의 픽셀값의 변화가 있는 곳을 볼 수 있음

dyy = dy*dy # 2차 미분
dxx = dx*dx # 2차 미분
dyx = dy*dx 

gdyy= cv.filter2D(dyy,cv.CV_32F,gausianfilter) # 가우시안 필터를 적용한 2차 모멘트 행렬의 원소
gdxx= cv.filter2D(dxx,cv.CV_32F,gausianfilter) # 가우시안 필터를 적용한 2차 모멘트 행렬의 원소
gdyx= cv.filter2D(dyx,cv.CV_32F,gausianfilter) # 가우시안 필터를 적용한 2차 모멘트 행렬의 원소

C = (gdyy*gdxx-gdyx*gdyx)-0.04*(gdyy+gdxx)*(gdyy+gdxx) # 지역 특징일 가능성이 있는 C값 계산 (고윳값을 통한)

for j in range(1,C.shape[0]-1):
    for i in range(1,C.shape[1]-1):
        if C[j,i] > 0.1 and sum(sum(C[j,i]>C[j-1:j+2,i-1:i+2]))==8:
            img[j,i]=9 # 특징점은 9로 치환
        '''sum(sum(C[j,i]>C[j-1:j+2,i-1:i+2])) 은 중심점과 그 이웃 8점과의 비교해서 8이면 특징점'''

np.set_printoptions(precision=2) # 소수점 이하 두 자리까지만 출력

popping = np.zeros([160,160],np.uint8)

for j in range(0,160):
    for i in range(0,160):
        popping[j,i]=np.uint8((C[j//16,i//16]+0.06)*700)
        

cv.imshow("Image Display",popping)
cv.waitKey()
cv.destroyAllWindows()