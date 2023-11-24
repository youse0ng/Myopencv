import cv2
import os
import numpy as np

path = 'local_featuers/'
filePath = os.path.join(path, "soccer.mp4")
print(filePath)

if os.path.isfile(filePath):	# 해당 파일이 있는지 확인
    # 영상 객체(파일) 가져오기
    cap = cv2.VideoCapture(filePath)
else:
    print("파일이 존재하지 않습니다.")  

# 프레임을 정수형으로 형 변환
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))	# 영상의 넓이(가로) 프레임
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))	# 영상의 높이(세로) 프레임
 
frame_size = (frameWidth, frameHeight)
print('frame_size={}'.format(frame_size))

frameRate = 33
def create_keypoint_descriptor(image_path) -> tuple:
    '''이미지의 keypoint와 이미지의 descriptor 추출'''
    img=cv2.imread(image_path)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    kp,des=sift.detectAndCompute(gray,None)
    return (kp,des) 
img1=cv2.imread('local_featuers\data\9.jpg')[220:250,370:395] # 검출할려는 모델 영상
gray=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
sift=cv2.SIFT_create()
kp_model,des_model=sift.detectAndCompute(gray,None)

print(f'모델 장면 특징점 개수: {len(kp_model)}')
flann_matcher=cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
T=0.88 # 임계값
while True:
    # 한 장의 이미지(frame)를 가져오기
    # 영상 : 이미지(프레임)의 연속 
    # 정상적으로 읽어왔는지 -> retval
    # 읽어온 프레임 -> frame
    retval, frame = cap.read()
    if not(retval):	# 프레임정보를 정상적으로 읽지 못하면
        break  # while문을 빠져나가기
    gray1=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    kp1,des1=sift.detectAndCompute(gray1,None)
    knn_match=flann_matcher.knnMatch(des_model,des1,2)
    good_match=[]
    for nearest1,nearest2 in knn_match:
        if (nearest1.distance/nearest2.distance)<T:
            good_match.append(nearest1)
    img_match=np.empty((max(img1.shape[0],frame.shape[0]),img1.shape[1]+frame.shape[1],3),dtype=np.uint8)
    cv2.drawMatches(img1,kp_model,frame,kp1,good_match,img_match,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('img_match',img_match)
    key = cv2.waitKey(frameRate)  # frameRate msec동안 한 프레임을 보여준다
    # 키 입력을 받으면 키값을 key로 저장 -> esc == 27(아스키코드)
    if key == 27:
        break	# while문을 빠져나가기
        
if cap.isOpened():	# 영상 파일(카메라)이 정상적으로 열렸는지(초기화되었는지) 여부
    cap.release()	# 영상 파일(카메라) 사용을 종료
    