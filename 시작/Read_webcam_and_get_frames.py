import cv2 as cv
import numpy as np
import sys

cap=cv.VideoCapture(0,cv.CAP_DSHOW) # 카메라와 연결시도

if not cap.isOpened():
    sys.exit("카메라 연결 실패")

frames=[]
while True:
    ret,frame=cap.read()

    if not ret:
        print("프레임 획득에 실패했습니다.")
        break

    cv.imshow("Video display",frame)

    key=cv.waitKey(1)
    if key == ord("c"): # 'c'키를 누를때마다 frame 저장
        frames.append(frame)
    elif key == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

if len(frames)>0: # 프레임이 저장되었다면,
    imgs=frames[0]
    for i in range(1,min(3,len(frames))): # min(3,1)=1 || min(3,5)=3, 즉, 최대 3장까지만 출력하겠다.
        imgs=np.hstack((imgs,frames[i])) # frame의 3차원 배열을 옆으로 stack하여 저장함.

    cv.imshow("collecte images",imgs)

    cv.waitKey()
    cv.destroyAllWindows()