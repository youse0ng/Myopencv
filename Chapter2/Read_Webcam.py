import cv2 as cv
import sys

cap=cv.VideoCapture(0,cv.CAP_DSHOW) # 카메라와 연결시도 
print(f'cap:{cap}')
# 웹캠이 1개인 경우엔 0을 준다.
if not cap.isOpened():
    sys.exit('카메라 연결 실패')
while True:
    ret,frame=cap.read() # 비디오를 구성하는 프레임 획득 
    # ret 객체에 프레임 획득에 대한 성공 여부를 저장
    print(ret)
    if not ret:
        print("프레임 획득에 실패하여 루프를 나갑니다.")
        break
    
    cv.imshow('Video display',frame)

    key=cv.waitKey(1)  # 1밀리초 동안 키보드 입력을 기다림
    if key==ord('q'): # q키가 들어오면 루프를 빠져나옴
        break

cap.release() # 카메라와의 연결을 끊음
cv.destroyAllWindows()
