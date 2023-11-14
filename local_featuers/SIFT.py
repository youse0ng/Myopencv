import cv2 as cv

img=cv.imread('local_featuers\mot_color70.jpg')
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

sift=cv.SIFT_create(contrastThreshold=0.1,
                    sigma=1.6,
                    edgeThreshold=10)
keypoint,descriptor=sift.detectAndCompute(gray,None) # 키포인트 검출과 디스크립터 검출
 
gray = cv.drawKeypoints(gray,keypoint,None,flags=cv.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
cv.imshow('sift',gray)

img2 =cv.imread('local_featuers\mot_color83.jpg')
gray2 = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)

keypoint2,descriptor2=sift.detectAndCompute(gray2,None)
gray2 = cv.drawKeypoints(gray2,keypoint2,None,flags=cv.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
cv.imshow('sift2',gray2)

cv.waitKey()
cv.destroyAllWindows()

'''개인적으로 SIFT 알고리즘은 API를 통한 코딩의 이해보단 알고리즘의 이해가 더 중요하기 때문에
   여기서 적지 않겠습니다.

   중요한 개념은 Keypoint 검출과 Descriptor 기술자인데,
   keypoint는 다중 scale 영상을 추출하므로 scale에 robust한 효과를 가져옵니다.

   Keypoint는 다중 scale 영상, 즉 영상이미지에 가우시안 스무딩을 (sigma) 표준편차를 증가하여 구합니다.

   원래는 가우시안 필터가 적용된 영상에 Laplacian of Gaussian을 컨볼루션 적용하지만, 이는 비용이 많이들어
   DOG(difference of Gaussian)을 이용하여 LoG와 유사한 영상을 얻음으로써 비용을 줄입니다.

   이때, DOG 영상들이 많이 생성되는데, i-1번째,i번째,i+1번째의 DOG 영상을 비교하여 DOG값이 큰 극점을 구합니다.
   이때 큰 극점은 특징점이라고 판단됩니다.

   허나 스케일과 극점의 위치를 가지고는, 매칭을 시키기에 많이 부족하여 Descriptor(특징벡터)를 추출합니다.

   그리고, 특징점의 주변 픽셀들의 어떠한 그래디언트 강도와 그래디언트 방향을 구하여, histogram화(histogram의 bin 개수는 360//10)합니다.

   이때, 최댓값을 가지는 지배적인 방향 theta를 구합니다.

   또 theta 방향을 기준으로 window를 씌워서 가우시안 필터 거쳐(주변의 픽셀 magnitude는 낮아지고 중심의 magnitude는 커지는 효과)

   16 * 16 사이즈의 작은 영역을 얻고 4*4 크기의 영역으로 16개의 블록을 얻는다.

   각 블록은 자신이 속한 16개의 화소의 그래디언트 방향을 얻고 8단계로 방향을 양자화하고 히스토그램을 구한다.

   이때, 만들어지는 16개(블록) * 8차원 히스토그램정보(그래디언트 방향이 담김)
   16*8 =128차원의 기술자를 얻는다. (Descriptor) 
'''
