import torch
import torch.nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import torch.optim
train_data=datasets.MNIST('DeepLearningVision\data',True,transforms.ToTensor(),
                       None,True)

test_data=datasets.MNIST('DeepLearningVision\data',False,transforms.ToTensor(),
                       None,True)

classes=train_data.classes

print(f'train_data의 개수: {len(train_data)}')
print(f'test_data의 개수: {len(test_data)}')
print(f'train_data[0]: {train_data[0]}') # 튜플 (배열,클래스넘버)
print(f'train_data[0].shape:{train_data[0][0].shape}') # 배열의 shape
print(f'train_data[0][0]의 자료형: {type(train_data[0][0])}') # torch.Tensor형

train_dataloader=DataLoader(dataset=train_data,
                            batch_size=32,
                            shuffle=True)

test_dataloader=DataLoader(dataset=test_data,
                           shuffle=False,
                           batch_size=32)

print(dir(train_dataloader))
print(f'train_dataloader: {next(iter(train_dataloader))[0].shape}')
print(f'train_dataloader의 개수: {len(train_dataloader)}') # 60000 // 32

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
    
model_0=MNISTModel(input_shape=28*28,
                   hidden_units=512,
                   output_shape=10)

optimizer=torch.optim.SGD(params=model_0.parameters(),lr=0.01)
loss_fn=torch.nn.MSELoss()

EPOCH=10

for Batch,(x_train,y_train) in enumerate(train_dataloader):
    train_loss=0
    # Do forward pass (로짓 계산)
    y_logits=model_0(x_train)
    print(y_logits)

    # Calculate loss (기울기 계산)
    loss=loss_fn(y_logits,y_train)
    train_loss+=loss
    # optimizer zero grad (경사 추적 x)
    optimizer.zero_grad()
    # loss backward (역전파)
    loss.backward()
    # optimizer step (가중치 갱신)
    optimizer.step()