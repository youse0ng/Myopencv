# 이진화에 대한 수식
# b(j,i) ={
#             1, f(j,i)>=T
#             0, f(j,i)<T
#         }
'''이때 T는 Threshhold 임계값이고 보통 히스토그램의 계곡 근처를 임계값으로 결정하여 쏠림현상을 누그러뜨린다.'''
import cv2 as cv
import matplotlib.pyplot as plt

img=cv.imread("ImageProcessing\soccer.jpg")
h=cv.calcHist([img],[2],None,[256],[0,256]) # BGR 채널 중 R채널의 히스토그램을 구함 (빈도수)
plt.plot(h,color='r',linewidth=1)
plt.show()

# 히스토그램을 보면 계곡이 두군데 나온다. 100과 150 어디를 중점으로 해서 T를 설정해야하는가??

# 어디를 설정해야 T를 잘 정한 것인지 알 수 없기 때문에, 오츄 알고리즘이 탄생함.
T=100
bin_img=img[:,:,2] # R channel 
print(bin_img)
print(bin_img.shape)
for col in range(0,948):
    for row in range(0,1434):
        if bin_img[col][row] >= T:
            bin_img[col][row]=1
        else: 
            bin_img[col][row]=0
print(bin_img)
plt.imshow(bin_img)
plt.show()