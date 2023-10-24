'''
히스토그램 평활화는 히스토그램이 평평하게 되도록 영상을 조작해 영상의 명암을 대비를 높이는 기법이다.

명암 대비가 높아지면 영상에 있는 물체를 더 잘 식별할 수 있다.

히스토그램 평활화의 수식
l' = round(h"(l) * (L-1))


l은 명암값, h"는 누적 정규화 히스토그램이다.
h"는 h' 정규화 히스토그램으로 구한다.
'''

import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread("ImageProcessing\mistyroad.jpg")

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY) # 명암 영상으로 변환하고 출력
plt.imshow(gray,cmap='gray'),plt.xticks([]),plt.yticks([]),plt.show()

h = cv.calcHist([gray],[0],None,[256],[0,256]) # 명암 영상인 gray의 히스토그램 계산하고 plot
plt.plot(h,color='r',linewidth=1),plt.show()

equal=cv.equalizeHist(gray) # equalization 함수를 통해 gray 이미지 => 히스토그램 평활화를 진행한 명암 영상
plt.imshow(equal,cmap='gray'),plt.xticks([]),plt.yticks([]),plt.show()

h=cv.calcHist([equal],[0],None,[256],[0,256]) # 히스토그램 평활화를 진행한 영상의 히스토그램 계산
plt.plot(h,color='r',linewidth=1),plt.show()


# 히스토그램의 평활화를 함수로 만들기
def histogram_equalization(img_file) -> None:
    img = cv.imread(img_file)
    
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY) # 명암 영상으로 변환하고 출력
    plt.imshow(gray,cmap='gray'),plt.xticks([]),plt.yticks([]),plt.show()

    h = cv.calcHist([gray],[0],None,[256],[0,256])
    plt.plot(h,color='r',linewidth=1),plt.show()

    equal=cv.equalizeHist(gray)
    plt.imshow(equal,cmap='gray'),plt.xticks([]),plt.yticks([]),plt.show()

    h=cv.calcHist([equal],[0],None,[256],[0,256])
    plt.plot(h,color='r',linewidth=1),plt.show()

# 내가 준비하는 데이터 이미지는 어떻게 반응 할까?
histogram_equalization("ImageProcessing\A1C_20220818_000018.jpg")
