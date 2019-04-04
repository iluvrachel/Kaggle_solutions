import numpy as np 
import pandas as pd 
import torch
from torch import nn
# from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader
import os
from tqdm import tqdm
import scipy.ndimage

train_data_dir = "/home/zhizhao/Earthquake/train.csv"

def read_data(data_path):

    data = pd.read_csv(data_path,dtype={"acoustic_data": np.float32, "time_to_failure": np.float32})
    return data

def extract_features(df):
    x = df.acoustic_data.values

    mean = np.mean(x)
    # median = np.median(x)
    standard = np.std(x)
    # max = np.max(x)
    
    x_ds = scipy.ndimage.zoom(x,0.5)
    mean_ds = np.mean(x_ds)
    standard_ds = np.std(x_ds)

    return np.c_[mean,standard,mean_ds,standard_ds]



class TrainData(Dataset):
    def __init__(self, df, window_size=1000, sequence_len=150):
        self.rows = df.shape[0] // (window_size*sequence_len)
        self.data, self.labels = [], []
        for segment in range(self.rows):
            seg = df.iloc[segment*window_size*sequence_len: (segment+1)*window_size*sequence_len]
            # x = seg.acoustic_data.values
            y = seg.time_to_failure.values[-1]
            X_feature = extract_features(seg) 
            self.data.append(X_feature)
            self.labels.append(y)
    
    def __len__(self):
        return self.rows
    
    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.data[idx].astype(np.float32)),
            self.labels[idx]
        )


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1) # one output
    
    def forward(self, x):

        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        c0 = torch.zeros(1, x.size(0), self.hidden_size)
        
        out, _ = self.lstm(x, (h0,c0))
        
        # output the last step
        out = self.fc(out[:, -1, :])
        return out.view(-1)

origin_data = read_data(train_data_dir)
train_data = TrainData(origin_data)

batch_size = 100
n_steps = len(train_data) // batch_size

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)

n_features = extract_features(origin_data[0:150000]).shape[1]

input_size = n_features
hidden_size = 32
model = LSTM(input_size, hidden_size)

learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(2):
    for i, (data, labels) in enumerate(train_loader):
        outputs = model(data)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print("[Epoch {0}/2, Step {1}/{2}]  loss: {3}".format(epoch,i,n_steps,loss))

torch.save(model.state_dict(),'trained_model.pth')
# generate submission
model = LSTM(input_size, hidden_size)
model.load_state_dict(torch.load('trained_model.pth'))
model.eval()
submission = pd.read_csv('sample_submission.csv', index_col='seg_id', dtype={"time_to_failure": np.float32})

test_data_dir = "/home/zhizhao/Earthquake/test/"

for i,seg_id in enumerate(tqdm(submission.index)):
    seg = pd.read_csv(test_data_dir + seg_id + '.csv')
    # x = seg['acoustic_data'].values
    x_f = torch.from_numpy(extract_features(seg))
    x_f = x_f.reshape(-1,1,n_features)
    x_f = x_f.float()
    # print(x_f.shape)
    submission.time_to_failure[i] = model(x_f)

submission.to_csv('submission.csv')
