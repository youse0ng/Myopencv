import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("ImageProcessing\JohnHancocksSignature.png",cv.IMREAD_UNCHANGED)
print(img.shape)
t,bin_img=cv.threshold(img[:,:,3],0,255,cv.THRESH_BINARY+cv.THRESH_OTSU) # 오츄 알고리즘을 통한 Thershold 이진화 
plt.imshow(bin_img,cmap='gray'),plt.xticks([]),plt.yticks([])
plt.show()

b=bin_img[bin_img.shape[0]//2:bin_img.shape[0],0:bin_img.shape[0]//2+1]
plt.imshow(b,cmap='gray'),plt.xticks([]),plt.yticks([])
plt.title('잘라낸 영상')
plt.show()

# 구조 요소
se=np.uint8([[0,0,1,0,0],
             [0,1,1,1,0],
             [1,1,1,1,1],
             [0,1,1,1,0],
             [0,0,1,0,0]])

b_dilation=cv.dilate(b,se,iterations=1)
plt.imshow(b_dilation,cmap='gray'),plt.xticks([]),plt.yticks([]) # 팽창
plt.title('팽창')
plt.show()

b_erosion=cv.erode(b,se,iterations=1)
plt.imshow(b_erosion,cmap='gray'),plt.xticks([]),plt.yticks([]) # 침식
plt.title('침식')
plt.show()

b_closing=cv.erode(cv.dilate(b,se,iterations=1),se,iterations=1) 
plt.imshow(b_closing,cmap='gray'),plt.xticks([]),plt.yticks([]) # 닫기 
plt.title('닫기')
plt.show()