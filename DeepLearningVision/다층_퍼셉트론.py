import torch
import torch.nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import torch.optim
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# MNIST 데이터셋 불러오기
train_data=datasets.MNIST('DeepLearningVision\data',True,transforms.ToTensor(),
                       None,True)

test_data=datasets.MNIST('DeepLearningVision\data',False,transforms.ToTensor(),
                       None,True)

# 데이터셋의 클래스 객체 저장
classes=train_data.classes

# train_data 데이터셋 정보를 확인 (dir(train_data))
print(f'train_data의 개수: {len(train_data)}')
print(f'test_data의 개수: {len(test_data)}')
print(f'train_data[0]: {train_data[0]}') # 튜플 (배열,클래스넘버)
print(f'train_data[0].shape:{train_data[0][0].shape}') # 배열의 shape
print(f'train_data[0][0]의 자료형: {type(train_data[0][0])}') # torch.Tensor형

# 데이터셋을 BATCH_SIZE (SGD:Stochastic Gradient Descent) 스토캐스택 경사하강법을 하기 위한 DataLoader 생성
train_dataloader=DataLoader(dataset=train_data,
                            batch_size=32,
                            shuffle=True)

test_dataloader=DataLoader(dataset=test_data,
                           shuffle=False,
                           batch_size=32)
print(f'torch_version:{torch.__version__}')
# dataloader의 기능 function 확인 (class method, attribution)
print(dir(train_dataloader))
print(f'train_dataloader: {next(iter(train_dataloader))[0].shape}')
print(f'train_dataloader의 개수: {len(train_dataloader)}') # 60000 // 32

# Train_dataloader를 학습하기 위한 딥러닝 신경망 모델 생성
# Fully Connected Layer 2개 생성
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

# 28 * 28 사이즈의 이미지 크기이기에 픽셀 하나를 노드 하나로 취급 
# hidden unit 512개의 노드
# output_shape은 클래스가 10개 있으므로, 출력층 노드도 10개로 지정
model_0=MNISTModel(input_shape=28*28,
                   hidden_units=512,
                   output_shape=10)

# 최적화 기법 (SGD) 정의
# 최적화할 손실함수 정의 MSELoss : Mean Squared Loss (평균 제곱 오차)
optimizer=torch.optim.SGD(params=model_0.parameters(),lr=0.1) # model_0의 파라미터를 갱신, learning rate는 0.01 (가중치 - 0.01 * Loss에 대한 해당 가중치의 Gradient)
loss_fn=torch.nn.CrossEntropyLoss()

# 모델 학습하는데 걸리는 시간 측정을 위한 함수
from timeit import default_timer as timer
def print_train_time(start:float,
                     end: float,
                     device:torch.device="cuda"):
  """Prints difference between start and end time."""
  total_time=end-start
  print(f"{device}상에서 걸린 학습시간: {total_time:.3f} seconds")
  return total_time

# acc 평가
def accuracy_fn(prediction,true):
    correct=torch.eq(prediction,true).sum().item()
    acc=(correct/len(true))*100
    return acc


start_time=timer()
train_acc_list=[]
train_loss_list=[]
test_loss_list=[]
test_acc_list=[]
EPOCH=10 # 60000개의 데이터에 대한 훑고감 정도? 10번 시행
for epoch in tqdm(range(EPOCH)):
    train_loss=0
    train_acc=0
    for Batch,(x_train,y_train) in enumerate(train_dataloader):
        model_0.train(True)
        # Do forward pass (로짓 계산)
        y_logits=model_0(x_train)
        # Calculate loss (손실 계산)
        loss=loss_fn(y_logits,y_train)
        train_loss+=loss
        # acc 계산
        train_acc+=accuracy_fn(y_logits.argmax(dim=1),y_train)
        # optimizer zero grad (경사 추적 x)
        optimizer.zero_grad()
        # loss backward (역전파를 통한 gradient 구하기)
        loss.backward()
        # optimizer step (가중치 갱신)
        optimizer.step()
    train_loss/=len(train_dataloader)
    train_acc/=len(train_dataloader)
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    print(f' train_loss: {train_loss}')
    # Evaluation
    test_loss=0.0
    test_acc=0.0
    model_0.eval()
    with torch.inference_mode():
        for X_test,y_test in test_dataloader:
            # Forward Pass
            test_logits=model_0(X_test)
            # 손실 계산
            test_loss+=loss_fn(test_logits,y_test)
        test_acc+=accuracy_fn(test_logits.argmax(dim=1),y_test)
        test_loss /= len(train_dataloader)
        test_acc_list.append(test_acc)
        test_loss_list.append(test_loss)
    print(f"Train_Loss: {train_loss:.2f}, Train_acc: {train_acc:.2f}, Test_Loss: {test_loss:.2f}, Test_Acc: {test_acc:.2f}")
end_time=timer()
print_train_time(start_time,end_time)

# 결과 플로팅 하고싶다.
plt.figure()
plt.plot(np.arange(EPOCH),train_acc_list,label='train_acc',color='green')
plt.plot(np.arange(EPOCH),test_acc_list,label='test_acc',color='red')
plt.xlabel('EPOCHS')
plt.ylabel("Accuracy")
plt.legend()

plt.figure()
plt.plot(np.arange(EPOCH),test_loss_list,label='test_loss')
plt.plot(np.arange(EPOCH),train_loss_list,label='train_loss',color='blue')
plt.xlabel('EPOCHS')
plt.ylabel("Loss")
plt.legend()
plt.show()