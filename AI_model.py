import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, accuracy_score

X_train = pd.read_csv('X_train_dt.csv')
y_train = pd.read_csv('y_train_dt.csv').squeeze()
X_validation = pd.read_csv('X_validation_dt.csv')
y_validation = pd.read_csv('y_validation_dt.csv').squeeze()

non_numeric_columns = X_train.select_dtypes(include=['object']).columns
X_train[non_numeric_columns] = X_train[non_numeric_columns].apply(pd.to_numeric, errors='coerce')
X_validation[non_numeric_columns] = X_validation[non_numeric_columns].apply(pd.to_numeric, errors='coerce')

X_train.fillna(X_train.mean(), inplace=True)
X_validation.fillna(X_validation.mean(), inplace=True)

X_train = X_train.values.astype(np.float32)
y_train = y_train.values.astype(np.int64)
X_validation = X_validation.values.astype(np.float32)
y_validation = y_validation.values.astype(np.int64)

X_train_tensor = torch.tensor(X_train)
y_train_tensor = torch.tensor(y_train)
X_validation_tensor = torch.tensor(X_validation)
y_validation_tensor = torch.tensor(y_validation)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
validation_dataset = TensorDataset(X_validation_tensor, y_validation_tensor)
validation_loader = DataLoader(validation_dataset, batch_size=64, shuffle=False)
import torch
import torch.nn as nn
import torch.nn.functional as F
class MultiheadAttention(nn.Module):

    def __init__(self, hid_dim, n_heads, dropout):
        super(MultiheadAttention, self).__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads

        assert hid_dim % n_heads == 0
        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)
        self.fc = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads]))

    def forward(self, query, key, value, mask=None):
        bsz = query.shape[0]
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)
        attention = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)
        attention = self.do(torch.softmax(attention, dim=-1))
        x = torch.matmul(attention, V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))
        x = self.fc(x)
        return x
class PositionwiseFeedforward(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(hid_dim, pf_dim)
        self.fc2 = nn.Linear(pf_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout):
        super().__init__()
        self.self_attn = MultiheadAttention(hid_dim, n_heads, dropout)
        self.layer_norm1 = nn.LayerNorm(hid_dim)
        self.pf = PositionwiseFeedforward(hid_dim, pf_dim, dropout)
        self.layer_norm2 = nn.LayerNorm(hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        _src = self.self_attn(src, src, src, src_mask)
        src = self.layer_norm1(src + self.dropout(_src))
        _src = self.pf(src)
        src = self.layer_norm2(src + self.dropout(_src))
        return src
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class Chomp1d(nn.Module):
    def __init__(self, padding):
        super(Chomp1d, self).__init__()
        self.padding = padding
    def forward(self, x):
        return x[:, :, :-self.padding]  

class TCN(nn.Module):
    def __init__(self, num_inputs,hid_dim, n_heads, num_channels, pf_dim, kernel_size=2, dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], 2) 
        self.attention = EncoderLayer(hid_dim=hid_dim, n_heads=n_heads, dropout=dropout,pf_dim = pf_dim)

    def forward(self, x, mask=None):
        x = x.unsqueeze(1)
        x = self.attention(x, mask)
        x = x.transpose(1, 2)
        y = self.network(x)
        o = self.linear(y[:, :, -1]) 
        return o


model = TCN(num_inputs=102, num_channels=[25, 50, 100],hid_dim=102, n_heads=2,pf_dim = 2048)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def validate_model(loader):
    model.eval()
    total_loss, total_accuracy, total_f1 = 0, 0, 0
    total_samples = 0
    for data, target in loader:
        output = model(data)
        loss = criterion(output, target)
        total_loss += loss.item() * data.size(0)
        _, predicted = torch.max(output, 1)
        total_accuracy += (predicted == target).sum().item()
        total_f1 += f1_score(target.numpy(), predicted.numpy(), average='weighted') * data.size(0)
        total_samples += data.size(0)
    avg_loss = total_loss / total_samples
    avg_accuracy = total_accuracy / total_samples
    avg_f1 = total_f1 / total_samples
    return avg_loss, avg_accuracy, avg_f1

def train_model(epochs):
    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        train_loss, train_accuracy, train_f1 = validate_model(train_loader)
        val_loss, val_accuracy, val_f1 = validate_model(validation_loader)
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train F1: {train_f1:.4f}')
        print(f'Epoch {epoch+1}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Validation F1: {val_f1:.4f}')

train_model(100)
