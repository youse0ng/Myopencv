import cv2 as cv

img = cv.imread("Edge_and_Region\\apples.jpg")
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# 케니 엣지 후 허브 변환을 통한 사과 검출.
canny1=cv.Canny(gray,80,240)

apples_after_canny = cv.HoughCircles(canny1,cv.HOUGH_GRADIENT,1,200,param1=80,param2=23.0,
                                     minRadius=50,maxRadius=119)

for i in apples_after_canny[0]:
    cv.circle(img,center=(int(i[0]),int(i[1])),radius=int(i[2]),color=(255,0,0),thickness=2)
# i[0]은 (x-a)**2에서 x에 해당, i[1]은 (y-b) **2 에서 y해당, radius는 i[2]이다.
print(apples_after_canny)
print(apples_after_canny.shape)
print(len(apples_after_canny[0]))

cv.imshow("Canny_apple",canny1)
cv.imshow("Apples",img)
cv.imwrite("Canny_apples.jpg",canny1)
cv.imwrite("Apples_after_canny_hough_transform.jpg",img)
cv.waitKey()
cv.destroyAllWindows()