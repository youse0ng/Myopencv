import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import winsound
import torch
from torchvision.io import read_image,ImageReadMode
from torchvision.transforms import Resize,Compose,Grayscale,ToTensor
from PIL.Image import open
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


class MNISTModel(torch.nn.Module):
    def __init__(self,
                 input_shape:int,
                 hidden_units:int,
                 output_shape:int):
        super().__init__()
        self.layer1=torch.nn.Sequential(
                            torch.nn.Flatten(),
                            torch.nn.Linear(in_features=input_shape,
                                                   out_features=hidden_units),
                            torch.nn.Linear(in_features=hidden_units,
                                                   out_features=output_shape)
        )
    def forward(self,x):
        return self.layer1(x)

# 모델 Parameter Load
model=MNISTModel(28*28,512,10)
model.load_state_dict(torch.load('DeepLearningVision\MNIST\models\model_0.pth'))
model.eval()
# 이미지 변환
test_image=open('DeepLearningVision\MNIST\\number5.jpg')
transformation=Compose([Grayscale(),Resize((28,28)),ToTensor()])
transformed_image=transformation(test_image)

classes=['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four', '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

# 변환된 이미지 확인
plt.figure()
plt.imshow(transformed_image.permute(1,2,0))
plt.title(f'Model_Predict: {classes[model(transformed_image).argmax(dim=1)]}')
plt.show()

def reset():
    global img
    
    img=np.ones((200,520,3),dtype=np.uint8)*255

    for i in range(5):
        cv.rectangle(img,(10+i*100,50),(10+(i+1)*100,150),(0,0,255))
    cv.putText(img,'e: Erase s:Show r:Recognition q:Quit',(10,40),cv.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0),1)

def grab_numerals():
    numerals=[]
    for i in range(5):
        roi=img[51:149,11+i*100:9+(i+1)*100,0]
        roi=255-cv.resize(roi,(28,28),interpolation=cv.INTER_CUBIC)
        numerals.append(roi)
    print(numerals)
    numerals=np.array(numerals)
    return numerals

def show():
    numerals=grab_numerals()
    plt.figure(figsize=(25,5))
    for i in range(5):
        plt.subplot(1,5,i+1)
        plt.imshow(numerals[i],cmap='gray')
        plt.xticks([]); plt.yticks([])
    plt.show()

def recognition():
    numerals=grab_numerals()
    numerals=numerals.reshape(5,784)
    numerals=numerals.astype(np.float32)/255.0
    for i in range(len(numerals)):
        result=model(numerals[i])
        class_id=np.argmax(result,dim=1)
        cv.putText(img,str(class_id[i]),(50+i*100,180),cv.FONT_HERSHEY_SIMPLEX,1,(255,0,0),1)
        winsound.Beep(1000,500)
    

BrushSiz=4
LColor=(0,0,0)

def writing(event,x,y,flags,param):
    if event==cv.EVENT_LBUTTONDOWN:
        cv.circle(img,(x,y),BrushSiz,LColor,-1)
    elif event == cv.EVENT_MOUSEMOVE and flags==cv.EVENT_FLAG_LBUTTON:
        cv.circle(img,(x,y),BrushSiz,LColor,-1)

reset()
cv.namedWindow('writing')
cv.setMouseCallback('writing',writing)

while(True):
    cv.imshow('writing',img)
    key=cv.waitKey(1)

    if key==ord('e'):
        reset()
    elif key == ord('s'):
        show()
    elif key == ord('r'):
        recognition()
    elif key == ord('q'):
        break

cv.destroyAllWindows()