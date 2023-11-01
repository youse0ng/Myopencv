import cv2 as cv

img = cv.imread("Edge_and_Region\soccer.jpg")

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

canny1=cv.Canny(gray,50,150) # 이력 임계값이 낮은 경우, low threshold :50, high threshold:150
canny2=cv.Canny(gray,100,200) # 이력 임계값이 높은 경우, low 100, high 200

print(canny1) # 0과 1로 이루어진 엣지 맵
print(canny1.shape)
print(type(canny1[0][0]))

cv.imshow("Original",gray)
cv.imshow("Canny1",canny1) # 등번호 3의 경계가 잘 따였지만, 잡초와 같은 곳에는 잡음이 심함
cv.imshow("Canny2",canny2) # 등번호 3의 경계를 잘 못 따였지만, 잡초와 같은 곳에 잡음이 덜심함

cv.waitKey()
cv.destroyAllWindows()


'''Non_Maximum Suppression 비최대 억제:
해당 화소의 엣지방향의 수직인 화소가 해당 화소의 엣지 강도보다 적다면 해당 화소는 살아남는 엣지가 된다.

Double Threshold
실제 엣지가 아닌데, 엣지라고 판단하고자 하는 오류를 줄이기 위한 gadget

high Threshold 보다 높은 엣지를 strong edge
low Threshold 보다 낮은 엣지는 없앰

low와 high 중간에 있는 엣지를 weak edge라고 놓고
인접 화소에 strong edge가 있다면 살리고 없다면 삭제
'''