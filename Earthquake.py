import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import scipy.ndimage
from scipy import stats
from sklearn import preprocessing

torch.set_num_threads(4)
train = pd.read_csv("/home/zhizhao/Earthquake/train.csv", dtype={"acoustic_data": np.float32, "time_to_failure": np.float32})
train.head()
sample_freq = 100
plt.plot(train.acoustic_data.values[::sample_freq])
plt.plot(train.time_to_failure.values[::sample_freq]*100)

def extract_features(z):
    z_abs = np.abs(z)
    temp = np.c_[
        z.mean(axis=1), 
        z.min(axis=1),
        z.max(axis=1), 
        z.std(axis=1),
        np.percentile(np.abs(z), q=[1, 5, 50, 95, 99], axis=1).T,
        stats.kurtosis(z,axis=1),
        stats.skew(z,axis=1),
        z_abs.max(axis=1),
        z_abs.min(axis=1),
        z_abs.std(axis=1)
    ]
    # temp.reshape(1,-1)
    return temp

def create_X(x, window_size=1000, sequence_len=150):
    tmp = x.reshape(sequence_len, -1)
    '''
    fft = np.fft.fft(x).reshape(sequence_len, -1)
    fft = fft.real
    '''

    '''
    return np.c_[
        extract_features(tmp),
        extract_features(tmp[:, -window_size // 10:]),
        extract_features(tmp[:, -window_size // 100:])
    ]
    '''
    return extract_features(tmp)

n_features = create_X(train.acoustic_data.values[0:150000]).shape[1]
print(n_features)
class TrainData(Dataset):
    def __init__(self, df, window_size=1000, sequence_len=150):
        self.rows = df.shape[0] // (window_size*sequence_len)
        self.data, self.labels = [], []
        for segment in range(self.rows):
            seg = df.iloc[segment*window_size*sequence_len: (segment+1)*window_size*sequence_len]
            x = seg.acoustic_data.values
            y = seg.time_to_failure.values[-1]
            self.data.append(create_X(x))
            self.labels.append(y)
    
    def __len__(self):
        return self.rows
    
    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.data[idx].astype(np.float32)),
            self.labels[idx]
        )
train_data = TrainData(train)
batch_size = 100
n_steps = len(train_data) // 100
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        hidden = (
            torch.zeros(1, x.size(0), self.hidden_size),
            torch.zeros(1, x.size(0), self.hidden_size)
        )
        
        out, _ = self.lstm(x, hidden)
        
        out = self.fc(out[:, -1, :])
        return out.view(-1)
input_size = n_features
hidden_size = 50
model = LSTM(input_size, hidden_size)
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for epoch in range(15):
    for i, (data, labels) in enumerate(train_loader):
        outputs = model(data)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print("[Epoch {0}/2, Step {1}/{2}]  loss: {3}".format(epoch,i,n_steps,loss))
torch.save(model.state_dict(),'trained_model_test.pth')

# generate submission
model = LSTM(input_size, hidden_size)
model.load_state_dict(torch.load('trained_model_test.pth'))
model.eval()
submission = pd.read_csv('sample_submission.csv', index_col='seg_id', dtype={"time_to_failure": np.float32})

test_data_dir = "/home/zhizhao/Earthquake/test/"

for i,seg_id in enumerate(tqdm(submission.index)):
    seg = pd.read_csv(test_data_dir + seg_id + '.csv')
    x = seg['acoustic_data'].values
    # print(x.shape)
    x_f = create_X(x)
    x_f = x_f.reshape(-1,150,n_features)
    x_f = torch.from_numpy(x_f)
    x_f = x_f.float()
    # print(x_f.shape)
    submission.time_to_failure[i] = model(x_f)

submission.to_csv('T_submission.csv')
