import skimage
import numpy as np
import cv2 as cv
import time
import matplotlib.pyplot as plt

# 교재에 있는 skimage.future.graph는 skimage의 버전 0.22.0에서는 분리되었다. 그래서 skimage documentation에 맞게 구현하였다.

coffee=skimage.data.coffee()
print(f'coffee 사진의 픽셀 정보: {coffee}')
print(f'coffee 이미지의 Shape: {coffee.shape}')

start=time.time()
slic=skimage.segmentation.slic(coffee,compactness=20,n_segments=600,start_label=1) # slic알고리즘(슈퍼 픽셀)을 이용하여 이미지 세그멘테이션 (슈픽픽셀 개수는 600개)
print(f'slic 알고리즘 적용한 이미지 shape: {slic.shape}') 
print(f'slic image: {slic}') # 2d array return 

graph=skimage.graph.rag_mean_color(coffee,slic,mode='similarity') # similarity (유사도)를 엣지 가중치로 사용하여 그래프를 구성하여 g에 저장
ncut=skimage.graph.cut_normalized(slic,graph)	# 정규화 절단, 정규화 절단은 super pixel을 이용한 노드를 사용
print(coffee.shape,' Coffee 영상을 분할하는데 ',time.time()-start,'초 소요')

marking=skimage.segmentation.mark_boundaries(coffee,ncut)
ncut_coffee=np.uint8(marking*255.0)

cv.imshow('Normalized cut',cv.cvtColor(ncut_coffee,cv.COLOR_RGB2BGR))  

cv.waitKey()
cv.destroyAllWindows()