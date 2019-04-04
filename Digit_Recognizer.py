import torch
from torch import nn
from torch.autograd import Variable as V
import torch.nn.functional as F
import numpy as np
import pandas as pd 
import scipy 
import time
from tqdm import tqdm


IS_TRAINING = False
SUBMIT = True
torch.manual_seed(1)
EPOCH = 50
BATCH_SIZE = 50
LR = 0.001

TRAIN_DATA = "train.csv"

# torch.set_default_tensor_type("torch.DoubleTensor")

#-----Funtions
def to_image(data):
    data = data.view(-1,1,28,28)
    return data

def read_data(data_path):
    train = pd.read_csv(data_path)
    data = train.drop('label',axis=1)
    data = data.values
    label = train['label'].values
    
    x = data[1000:,:]
    x = torch.from_numpy(x).float()
    val_x = data[:1000,:]
    val_x = torch.from_numpy(val_x).float()
    y = label[1000:]
    y = torch.from_numpy(y).long()
    val_y = label[:1000]
    val_y = torch.from_numpy(val_y).long()
    return V(x),V(y),V(val_x),val_y

#-----Network Structure
class net(nn.Module):
    def __init__(self):
        super(net,self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(1,16,kernel_size=5,stride=1,padding=2),
                        nn.BatchNorm2d(16),
                        nn.ReLU(),
                        nn.MaxPool2d(2)
                        )
        self.conv2 = nn.Sequential(
                        nn.Conv2d(16,32,kernel_size=5,stride=1,padding=2),
                        nn.BatchNorm2d(32),
                        nn.ReLU(),
                        nn.MaxPool2d(2)
                        )
        self.fc = nn.Linear(32,10)

    def forward(self,x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.view(out.size(0),-1)
        out = self.fc(out)
        return out

class conv_net(nn.Module):

    def __init__(self):
        super(conv_net,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,10,5,1,1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.BatchNorm2d(10)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(10,20,5,1,1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.BatchNorm2d(20) # num_features is channels' number
        )
        self.fc1 = nn.Sequential(
            nn.Linear(500,60),
            nn.Dropout(0.5),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(60,20),
            nn.Dropout(0.5),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(20,10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1,500)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


#-----Prepare Work
train_data_x, train_data_y, val_data_x, val_data_y = read_data(TRAIN_DATA) 
train_data_x = to_image(train_data_x)
val_data_x = to_image(val_data_x)


if IS_TRAINING:
    model = conv_net().cuda()
    loss_function = nn.CrossEntropyLoss() # this should be define before usage
    optimizer = torch.optim.Adam(model.parameters(),lr=LR)

    for epoch in tqdm(range(EPOCH)):
        start = time.time()
        index = 0
        if epoch%100 == 0:
            for param_group in optimizer.param_groups:
                LR = LR * 0.95
                param_group['lr'] = LR

        for i in tqdm(range(int(len(train_data_x)/BATCH_SIZE)),total=int(len(train_data_x)/BATCH_SIZE)):
            batch_x = train_data_x[index:index+BATCH_SIZE]
            batch_y = train_data_y[index:index+BATCH_SIZE]
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            output = model(batch_x)
            loss = loss_function(output,batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            index += BATCH_SIZE # next batch
            # print(loss)

        duration = time.time()-start
        print('Training duration:%.4f'%duration)

    torch.save(model.state_dict(),'trained_model.pth')


#-----Validation
model = conv_net().cpu()
model.load_state_dict(torch.load('trained_model.pth'))
model.eval()
test_output = model(val_data_x)

pred_y = torch.max(test_output, 1)[1].data.squeeze()
result = pred_y - val_data_y
accuracy = float(sum(result == 0)) / float(val_data_y.size(0))
print('Val_Acc: %.4f'%accuracy)
        
#-----Generate Submission

if SUBMIT:
    submission = pd.read_csv("sample_submission.csv")
    model.eval()
    test = pd.read_csv('test.csv')
    test_data = torch.from_numpy(test.values).float()
    test_data = to_image(V(test_data))
    result = torch.Tensor()
    index = 0
    for i in tqdm(range(int(test_data.shape[0]/BATCH_SIZE)),total=int(test_data.shape[0]/BATCH_SIZE)):
        label_prediction = model(test_data[index:index+BATCH_SIZE])
        
        #print(result)
        #print("LABEL!!!")
        #print(label_prediction)
        if index == 0:
            result = label_prediction.clone()
        else:
            result = torch.cat((result,label_prediction),0) # concat two tensor on axis 0
        index += BATCH_SIZE

    _,submission['Label'] = torch.max(result.data,1) # cover the origin csv_file

    submission.to_csv("submission.csv",index=False)
