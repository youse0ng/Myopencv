import cv2 as cv
import sys

img = cv.imread("soccer.jpg") # 영상 읽기
# soccer.jpg가 이 프로그램이 저장되어 있는 폴더에 있어야 한다.

if img is None: # 이미지가 없다면
    print("이미지가 없습니다.")
    sys.exit("파일을 찾을 수 없습니다.") # 프로그램 종료

cv.imshow("Image Display",img) # 윈도우에 영상 표시
# Image Display는 윈도우 창의 제목, img 디스플레이할 영상
cv.waitKey()
cv.destroyAllWindows()

print(img)
print(type(img)) # OpenCV는 넘파이로 영상을 표현한다.
print(img.shape) # 3차원 배열