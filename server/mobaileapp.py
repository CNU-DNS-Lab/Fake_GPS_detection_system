from flask import send_file
import csv
import sqlite3
from flask import render_template
from datetime import datetime
import pandas as pd
from haversine import haversine
import numpy as np
import math
import pytz 
from dateutil import parser
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import pandas as pd
import numpy as np
import re
from sklearn.metrics import f1_score, accuracy_score
import torch.optim as optim
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.nn.functional as F

class MultiheadAttention(nn.Module):
    # n_heads：多头注意力的数量
    # hid_dim：每个词输出的向量维度
    def __init__(self, hid_dim, n_heads, dropout):
        super(MultiheadAttention, self).__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads

        # 强制 hid_dim 必须整除 h
        assert hid_dim % n_heads == 0
        # 定义 W_q 矩阵
        self.w_q = nn.Linear(hid_dim, hid_dim)
        # 定义 W_k 矩阵
        self.w_k = nn.Linear(hid_dim, hid_dim)
        # 定义 W_v 矩阵
        self.w_v = nn.Linear(hid_dim, hid_dim)
        self.fc = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        # 缩放
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads]))

    def forward(self, query, key, value, mask=None):
        # K: [64,10,300], batch_size 为 64，有 12 个词，每个词的 Query 向量是 300 维
        # V: [64,10,300], batch_size 为 64，有 10 个词，每个词的 Query 向量是 300 维
        # Q: [64,12,300], batch_size 为 64，有 10 个词，每个词的 Query 向量是 300 维
        bsz = query.shape[0]
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        # 这里把 K Q V 矩阵拆分为多组注意力，变成了一个 4 维的矩阵
        # 最后一维就是是用 self.hid_dim // self.n_heads 来得到的，表示每组注意力的向量长度, 每个 head 的向量长度是：300/6=50
        # 64 表示 batch size，6 表示有 6组注意力，10 表示有 10 词，50 表示每组注意力的词的向量长度
        # K: [64,10,300] 拆分多组注意力 -> [64,10,6,50] 转置得到 -> [64,6,10,50]
        # V: [64,10,300] 拆分多组注意力 -> [64,10,6,50] 转置得到 -> [64,6,10,50]
        # Q: [64,12,300] 拆分多组注意力 -> [64,12,6,50] 转置得到 -> [64,6,12,50]
        # 转置是为了把注意力的数量 6 放到前面，把 10 和 50 放到后面，方便下面计算
        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)

        # 第 1 步：Q 乘以 K的转置，除以scale
        # [64,6,12,50] * [64,6,50,10] = [64,6,12,10]
        # attention：[64,6,12,10]
        attention = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # 把 mask 不为空，那么就把 mask 为 0 的位置的 attention 分数设置为 -1e10
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)

        # 第 2 步：计算上一步结果的 softmax，再经过 dropout，得到 attention。
        # 注意，这里是对最后一维做 softmax，也就是在输入序列的维度做 softmax
        # attention: [64,6,12,10]
        attention = self.do(torch.softmax(attention, dim=-1))

        # 第三步，attention结果与V相乘，得到多头注意力的结果
        # [64,6,12,10] * [64,6,10,50] = [64,6,12,50]
        # x: [64,6,12,50]
        x = torch.matmul(attention, V)

        # 因为 query 有 12 个词，所以把 12 放到前面，把 5 和 60 放到后面，方便下面拼接多组的结果
        # x: [64,6,12,50] 转置-> [64,12,6,50]
        x = x.permute(0, 2, 1, 3).contiguous()
        # 这里的矩阵转换就是：把多组注意力的结果拼接起来
        # 最终结果就是 [64,12,300]
        # x: [64,12,6,50] -> [64,12,300]
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
        # self attention
        _src = self.self_attn(src, src, src, src_mask)
        # apply residual connection
        src = self.layer_norm1(src + self.dropout(_src))

        # positionwise feedforward
        _src = self.pf(src)
        # apply residual connection
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

def init_db():
    conn = sqlite3.connect('location.db')
    c = conn.cursor()
    c.execute()
    conn.commit()
    conn.close()

init_db()

def standardize_group(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data.values.reshape(-1, 1)).flatten()
from flask import Flask, request, jsonify


app = Flask(__name__)

def get_kst_now():
    tz = pytz.timezone('Asia/Seoul') 
    return datetime.now(tz) 

@app.route('/submit_location', methods=['POST'])
def submit_location():
    data = request.json
    device_id = data.get('device_id', 'unknown')
    latitude = data['latitude']
    longitude = data['longitude']
    timestamp = datetime.now(pytz.timezone('Asia/Seoul')) 

    date = timestamp.strftime('%m/%d/%Y')
    hour = timestamp.strftime('%H')
    seconds = str(int(timestamp.timestamp()))

    conn = sqlite3.connect('location.db')
    c = conn.cursor()
    c.execute('INSERT INTO locations (android_id, latitude, longitude, timestamp, date, hour, seconds) VALUES (?, ?, ?, ?, ?, ?, ?)', 
              (device_id, latitude, longitude, timestamp, date, hour, seconds))
    conn.commit()
    conn.close()

    return jsonify({"status": "success", "message": "Location data received and stored successfully."})


@app.route('/download_csv')
def download_csv():
    try:
        conn = sqlite3.connect('location.db')
        c = conn.cursor()
        c.execute('SELECT * FROM locations')
        data = c.fetchall()
        conn.close()
        
        csv_filename = 'locations_' + get_kst_now().strftime('%Y-%m-%d_%H-%M-%S') + '.csv'
        with open(csv_filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['ID', 'Latitude', 'Longitude', 'Android ID', 'Timestamp'])
            writer.writerows(data)
        
        return send_file(csv_filename, as_attachment=True, attachment_filename=csv_filename)
    except Exception as e:
        return str(e)  

@app.route('/view_data')
def view_data():
    conn = sqlite3.connect('location.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM locations')
    locations = c.fetchall()
    conn.close()
    return render_template('view_data.html', locations=locations)

from sklearn.metrics import accuracy_score, f1_score
def convert_to_ascii(s):
    return ''.join(str(ord(c)) for c in s)
def preprocess_data(data):

    data.drop(['accuracy_in_meters'], axis=1, inplace=True)
    data.drop(['altitude_in_meters'], axis=1, inplace=True)
    data['service_type'] = data['service_type'].apply(convert_to_ascii)
    data['order_id'] = data['order_id'].apply(convert_to_ascii)
    data['driver_status'] = data['driver_status'].apply(convert_to_ascii)
    data['date'] = pd.to_datetime(data['date'])  
    reference_date = pd.Timestamp('2000-01-01')  
    data['date'] = (data['date'] - reference_date).dt.days  
    data[['service_type', 'driver_status', 'latitude','longitude']] = data.groupby('order_id')[['service_type', 'driver_status', 'latitude','longitude']].transform(standardize_group)
    return data


def filter_windows(df, window_size):
    windows = []

    for start_idx in range(len(df) - window_size + 1):
        window = df.iloc[start_idx:start_idx + window_size]
        if window['order_id'].nunique() == 1:
            windows.append(window.values) 
    

    return np.array(windows)

@app.route('/preprocess_data', methods=['POST'])
def preprocess_data_route():
    conn = sqlite3.connect('location.db')
    df = pd.read_sql_query("SELECT * FROM locations", conn)
    conn.close()


    processed_data = preprocess_data(df)  
    return jsonify(processed_data.to_dict(orient='records'))  
def get_data_from_csv(order_id):
    df = pd.read_csv('test_input.csv')
    filtered_df = df[df['order_id'] == order_id]  
    return filtered_df


def preprocess_data_for_model(df):
    print("Running preprocess_data_for_model...")

    processed_data = preprocess_data(df)

    print("Type of processed_data:", type(processed_data))
    if isinstance(processed_data, np.ndarray):
        print("Shape of processed_data:", processed_data.shape)
    
    window_size = 50
    processed_data = filter_windows(processed_data, window_size)

    if isinstance(processed_data, np.ndarray):
        print("Shape after filter_windows:", processed_data.shape)
    
    try:
        processed_data = processed_data[:, :, :-1] 
        print("Shape after slicing:", processed_data.shape)
        
        tensor = np.delete(processed_data, 0, axis=2) 
        print("Shape after deleting column:", tensor.shape)
        
        tensor = torch.tensor(tensor.astype(np.float32))
        print("Final tensor shape:", tensor.shape)
    except Exception as e:
        print("Error processing data:", e)
        return None

    return tensor

model = TCN(num_inputs=102, num_channels=[25, 50, 100],hid_dim=102, n_heads=2,pf_dim = 2048)
model.load_state_dict(torch.load('TCN_modelS.pth'))
model.eval()

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:

            order_id = request.form['order_id']
            df = get_data_from_csv(order_id)
            input_tensor = preprocess_data_for_model(df)  # 预处理函数需自行定义

            if input_tensor is None:
                return render_template('predictions.html', predictions=None, order_id=order_id, error="No valid data found for the given order_id.")

            with torch.no_grad():
                output = model(input_tensor)
                predictions = ['Real' if pred > 0 else 'fake' for pred in output.tolist()]
                print("Predictions:", predictions)
            return render_template('predictions.html', predictions=predictions, order_id=order_id)
        except Exception as e:
            print("Error:", e)
            return jsonify({'error': str(e)}), 500
    else:
        return render_template('predictions.html', predictions=None)









if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)


