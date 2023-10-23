import cv2 as cv
import sys

img = cv.imread("Chapter2\girl_laughing.jpg")

if img is None:
    sys.exit("파일을 찾을 수 없습니다.")

def draw(event,x,y,flags,param): # 콜백 함수 (매개 변수: event: 이벤트의 종류, x,y는 이벤트가 일어난 순간의 커서 위치)
    if event==cv.EVENT_LBUTTONDOWN: # 마우스 왼쪽 버튼 클릭 했을 때
        cv.rectangle(img,(x,y),(x+200,y+200),(0,0,255),2)
        cv.putText(img,"Laughing",(x,y-24),cv.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
    elif event == cv.EVENT_RBUTTONDOWN: # 마우스 오른쪽 버튼 클릭 했을 때
        cv.rectangle(img,(x,y),(x+100,y+100),(255,0,0),2)

    cv.imshow("Drawing",img)

cv.namedWindow("Drawing") # Drawing이라는 이름의 윈도우 생성
cv.imshow("Drawing",img) 

cv.setMouseCallback("Drawing",draw) # 마우스 이벤트가 발생하면 draw라는 콜백함수를 호출하라고 등록
# 마우스 이벤트는 버튼 클릭하기, 버튼에서 손 놓기, 커서 이동, 휠돌리기를 하면 발생한다.
while(True):
    if cv.waitKey(1)==ord('q'):
        cv.destroyAllWindows()
        break