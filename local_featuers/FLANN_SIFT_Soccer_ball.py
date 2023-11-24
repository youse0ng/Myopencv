import cv2 as cv
import numpy as np
import time
import glob

def create_keypoint_descriptor(image_path) -> tuple:
    '''이미지의 keypoint와 이미지의 descriptor 추출'''
    img=cv.imread(image_path)
    gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    sift=cv.SIFT_create()
    kp,des=sift.detectAndCompute(gray,None)
    return (kp,des)

img1=cv.imread('local_featuers\data\9.jpg')[220:250,370:395] # 검출할려는 모델 영상
gray=cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
sift=cv.SIFT_create()
kp_model,des_model=sift.detectAndCompute(gray,None)

print(f'모델 장면 특징점 개수: {len(kp_model)}')

image_paths='local_featuers\data/*.jpg' 
images=[img for img in glob.glob(image_paths)] # 검출할려는 프레임들

start=time.time()
flann_matcher=cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)

T=0.88 # Match T임계값
count=0
for image in images:
    kp1,des1=create_keypoint_descriptor(image)
    knn_match=flann_matcher.knnMatch(des_model,des1,2)
    good_match=[]
    for nearest1,nearest2 in knn_match:
        if (nearest1.distance/nearest2.distance)<T:
            good_match.append(nearest1)
    img_match=np.empty((max(img1.shape[0],cv.imread(image).shape[0]),img1.shape[1]+cv.imread(image).shape[1],3),dtype=np.uint8)
    count+=1
    cv.drawMatches(img1,kp_model,cv.imread(image),kp1,good_match,img_match,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imshow('Good Matches',img_match)
    
print('매칭에 걸린 시간: ',time.time()-start)
k=cv.waitKey()
cv.destroyAllWindows()