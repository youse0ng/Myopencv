import cv2 as cv
from pprint import pprint
img = cv.imread("Edge_and_Region\soccer.jpg")  # BGR (numpy.ndarray) 자료형
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # BGR (numpy.ndarray) 자료형
grad_x = cv.Sobel(gray,cv.CV_32F,1,0,ksize=3) # 948 * 1434 (grad_x) kernel[[-1,0,1]
                                                                        #  [-2,0,2]
                                                                        #  [-1,0,1]]
grad_y = cv.Sobel(gray,cv.CV_32F,0,1,ksize=3) # kernel  [-1,-2,-1]
                                                     #  [0, 0, 0]
                                                     #  [1, 2, 1]

sobel_x = cv.convertScaleAbs(grad_x) # 절대값을 취한 양수 영상으로 변환
sobel_y = cv.convertScaleAbs(grad_y) # 절대값을 취한 양수 영상으로 변환 나중에 edge_strength 엣지강도 
edge_strength = cv.addWeighted(sobel_x,0.5,sobel_y,0.5,0)
pprint(sobel_x)
cv.imshow('Original',gray)
cv.imshow('Sobel_x',sobel_x) # 수직 방향의 엣지가 선명 # 엣지 강도 맵
cv.imshow('Sobel_y',sobel_y) # 수평 방향의 엣지가 선명 # 엣지 강도 맵
cv.imshow('edge_strength',edge_strength) # 엣지가 있을것이란 가능성: 엣지강도의 imshow() 엣지 강도맵

cv.waitKey()
cv.destroyAllWindows()

'''엣지 연산은 미분식과 관련되어있다.
기본 영상의 화소값에 이웃한 미분 u 필터를 적용하면,
화소값이 급변하는 곳, 만약 화소값이 급락하는 곳에는 u필터(-1,1)또는 (-1,0,1)가 적용되면 음수로 나오고,
급등하는 곳은 양수로 나올 것이다.
이렇게 해서 나온 필터 영상을 f'라고 한다면
f'의 0이 아닌 값이 존재하는 곳이 화소가 변하는 부분인 즉 엣지영역이라고 추측할 수 있다.
그러면 값이 있다면 무조건 edge인가 ?? 그런 것은 또 아니다.

1차 미분인 필터가 주는 의미는 에지 발생 여부뿐만 아니라 엣지가 어떤 방향으로 향하는가: - 또는 +에 대한 정보를 제공한다는 점에 있다.
1차 미분인 필터를 프레윗 연산 필터이고

더 나아가 영교차(자신은 0이면서 동시에 이웃한 화소값이 서로 부호가 교차되는 자신)를 찾는 2차 미분은
소벨 연산자이다.

edge_strenth(엣지 강도)를 구하는 수식은 프레윗 연산 또는 소벨 연산을 거친 fx' ** 2 + fy' ** 2의 제곱근이다.
엣지 강도는 그 화소가 엣지일 가능성을 말한다.
'''