import pandas as pd
import numpy as np
import sys
import pickle
import os
import torch
import torch_geometric
from torch_geometric.utils import dense_to_sparse
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv
import torch.nn as nn
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from regression import *
import random
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
import warnings
warnings.simplefilter("ignore")

def set_seed(seed=42):
    # Python
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # CUDA & cuDNN
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False     
    torch.backends.cudnn.enabled = False     
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'

set_seed(1)




stock_df1 = pd.read_csv('NASDAQ100.csv')
stock_list = stock_df1['Ticker'].tolist()
# stock_list = []


stock_sector = pd.read_csv(r'NASDAQ100.csv', encoding='latin1')[['Ticker', 'GICS Sector']]
#let stock_sector only contain the stock in stock_list
stock_sector = stock_sector[stock_sector['Ticker'].isin(stock_list)]
sector2num = {}
num = 0
for i in stock_sector['GICS Sector']:
    if i not in sector2num.keys():
        sector2num[i] = num
        num += 1
dict_sector = {}
for index, item in stock_sector.iterrows():
    ticker = item['Ticker']
    sector = sector2num[item['GICS Sector']]
    if ticker not in dict_sector.keys():
        dict_sector[ticker] = sector

#To run our DualGAT model, the valid_pred_ic.csv, valid_label.csv, pred.csv, label.csv is the output form running our Temporal Pre-training Model, the code for the temporal pre-training model can be seen in https://github.com/SaizhuoWang/quantbench by specifying the model parameter as multi_scale_rnn.
df_valid1 = pd.read_csv('/mlruns/385851665399949056/4e34999712ae443491da0992a9d72d23/artifacts/outputs/valid_pred_ic.csv', index_col=0)
valid_label1 = pd.read_csv('mlruns/385851665399949056/4e34999712ae443491da0992a9d72d23/artifacts/outputs/valid_label.csv', index_col=0)
df_valid2 = pd.read_csv('mlruns/385851665399949056/4e34999712ae443491da0992a9d72d23/artifacts/outputs/pred.csv', index_col=0)
valid_label2 = pd.read_csv('mlruns/385851665399949056/4e34999712ae443491da0992a9d72d23/artifacts/outputs/label.csv', index_col=0)


df_valid = pd.concat([df_valid1,df_valid2])
valid_label = pd.concat([valid_label1,valid_label2])
df_valid = df_valid.stack().reset_index()
df_valid.columns = ['datetime', 'stock', 'valid_value']

valid_label = valid_label.stack().reset_index()
valid_label.columns = ['datetime', 'stock', 'label_value']

file_stock = 'psudo_combine_all.csv'
stock_df = pd.read_csv(file_stock)
stock_df = stock_df[stock_df['stock'].isin(stock_list)]

stock_df['psudo_label'] = np.where(stock_df['pseudo_gt'] == 'Bullish', 1, 0)
stock_df['gt_label'] = np.where(stock_df['gt_sentiment'] == 'Bullish', 1, 0)
stock_df.columns = ['stock', 'datetime', 'gt_sentiment', 'pseudo_gt', 'psudo_label', 'gt_label']
stock_df['datetime'] = pd.to_datetime(stock_df['datetime']).dt.date
df_valid['datetime'] = pd.to_datetime(df_valid['datetime']).dt.date
valid_label['datetime'] = pd.to_datetime(valid_label['datetime']).dt.date
print(df_valid)

stock_time_all = df_valid['datetime'].unique()
stock_time_all = sorted(stock_time_all)
time_mapping = {stock_time_all[i]: stock_time_all[i - 2] for i in range(2, len(stock_time_all))}

stock_df['datetime'] = stock_df['datetime'].map(time_mapping)

print(stock_df)

df_merged = pd.merge(df_valid, stock_df[['datetime', 'stock', 'psudo_label']],
                     on=['datetime', 'stock'], how='left')

max_date_merged = df_merged['datetime'].max()
max_date_valid = valid_label['datetime'].max()
df_merged_filtered = df_merged[df_merged['datetime'] != max_date_merged]
valid_label_filtered = valid_label[valid_label['datetime'] != max_date_valid]

label_notna = valid_label_filtered['label_value'].notna()
valid_label_filtered = valid_label_filtered[label_notna]

valid_notna = df_merged_filtered['valid_value'].notna()
df_merged_filtered = df_merged_filtered[valid_notna]

print(df_merged_filtered)
print(valid_label_filtered)

stock_price_df = pd.read_csv('close.csv')
stock_price_df = stock_price_df.rename(columns={'Unnamed: 0': 'datetime'})
stock_list.append('datetime')
stock_price_df = stock_price_df[stock_list]

min_date = pd.to_datetime(valid_label_filtered['datetime']).min() - pd.Timedelta(days=30)
max_date = pd.to_datetime(valid_label_filtered['datetime']).max()
stock_price_df['datetime'] = pd.to_datetime(stock_price_df['datetime'])
stock_price_df.set_index('datetime', inplace=True)

return_ratio_df = pd.DataFrame(index=stock_price_df.index)

for stock in stock_price_df.columns:
    price_series = stock_price_df[stock]
    price_day_1 = price_series.shift(-1)
    price_day_2 = price_series.shift(-2)
    return_ratio = (price_day_2 - price_day_1) / price_day_1
    return_ratio_df[stock] = return_ratio

stock_price_df = return_ratio_df

filtered_stock_price_df = stock_price_df[(stock_price_df.index >= min_date) & (stock_price_df.index <= max_date)]

stock_price = filtered_stock_price_df.stack().reset_index()
stock_price.columns = ['datetime', 'stock', 'label_value']
df_merged_filtered['datetime'] = pd.to_datetime(df_merged_filtered['datetime'])
merged_df = pd.merge(df_merged_filtered, stock_price, on=['datetime', 'stock'], how='left')

merged_df = merged_df.sort_values(by=['datetime', 'stock'])
stock_price = stock_price.sort_values(by=['datetime', 'stock'])
merged_df['datetime'] = pd.to_datetime(merged_df['datetime'])
stock_price['datetime'] = pd.to_datetime(stock_price['datetime'])


def calculate_avg(row, df):
    stock = row['stock']
    current_date = row['datetime']
    stock_data = df[(df['stock'] == stock) & (df['datetime'] <= current_date)].tail(30)
    if row['psudo_label'] == 1:
        label_values = stock_data[stock_data['label_value'] > 0]['label_value']
    elif row['psudo_label'] == 0:
        label_values = stock_data[stock_data['label_value'] < 0]['label_value']
    else:
        return None
    if len(label_values) > 0:
        return label_values.mean()
    else:
        return None

output_folder = 'correlation_matrices'
os.makedirs(output_folder, exist_ok=True)

correlation_matrices_only100 = {}

unique_dates = merged_df['datetime'].unique()

for current_date in unique_dates:
    current_date = pd.Timestamp(current_date)
    past_30_days_data = stock_price[(stock_price['datetime'] <= current_date) &
                                    (stock_price['datetime'] >= current_date - pd.Timedelta(days=30))]
    pivot_data = past_30_days_data.pivot(index='datetime', columns='stock', values='label_value')
    correlation_matrix = pivot_data.corr()
    correlation_matrices_only100[current_date] = correlation_matrix
    correlation_matrix.to_csv(f'{output_folder}/correlation_matrix_{current_date}.csv')

with open(f'{output_folder}/correlation_matrices_only100.pkl', 'wb') as f:
    pickle.dump(correlation_matrices_only100, f)

print("All correlation matrices have been saved in the folder.")


# let merged_df only contain the stock in stock_list
merged_df = merged_df[merged_df['stock'].isin(stock_list)]
merged_df['calculated_avg'] = merged_df.apply(lambda row: calculate_avg(row, merged_df), axis=1)
merged_df.to_csv('merged_df_with_calculated_only100.csv', index=False)
csv_file_path = 'merged_df_with_calculated_only100.csv'
merged_df = pd.read_csv(csv_file_path)



with open('correlation_matrices/correlation_matrices_only100.pkl', 'rb') as f:
    correlation_matrices = pickle.load(f)
merged_df['datetime'] = pd.to_datetime(merged_df['datetime'])

output_folder = 'graph_matrices'
os.makedirs(output_folder, exist_ok=True)
graph_matrices_positive_dict = {}
graph_matrices_negative_dict = {}

for current_date in correlation_matrices:
    correlation_matrix = correlation_matrices[current_date]
    print(current_date)
    
    current_stocks = correlation_matrix.columns
    
    selected_stocks = merged_df[
        (merged_df['datetime'] == current_date) & 
        (~merged_df['psudo_label'].isna())  
        # &(merged_df['calculated_avg'] != 0)
    ]['stock'].values
    
    graph_matrix_positive = pd.DataFrame(0, index=current_stocks, columns=current_stocks)
    graph_matrix_negative = pd.DataFrame(0, index=current_stocks, columns=current_stocks)
    
    for stock in current_stocks:
        pos_corr_stocks = correlation_matrix[stock][correlation_matrix[stock] > 0]
        neg_corr_stocks = correlation_matrix[stock][correlation_matrix[stock] < 0]

        if stock in selected_stocks:
            high_pos_corr_stocks = pos_corr_stocks[pos_corr_stocks > 0.6]
            if len(high_pos_corr_stocks) > 0:
                num=len(high_pos_corr_stocks)
                print('selected_num',num)
                if len(high_pos_corr_stocks[high_pos_corr_stocks > 0.85]) > 20:
                    high_pos_corr_stocks = high_pos_corr_stocks[high_pos_corr_stocks > 0.85]
                else:
                    high_pos_corr_stocks = high_pos_corr_stocks.nlargest(20)
                high_pos_corr_stocks = high_pos_corr_stocks[high_pos_corr_stocks > 0.6]
                graph_matrix_positive.loc[stock, high_pos_corr_stocks.index] = 1
            
            high_neg_corr_stocks = neg_corr_stocks[neg_corr_stocks < -0.6]
            if len(high_neg_corr_stocks) > 0:
                if len(high_neg_corr_stocks[high_neg_corr_stocks < -0.8]) > 20:
                    high_neg_corr_stocks = high_neg_corr_stocks[high_neg_corr_stocks < -0.8]
                else:
                    high_neg_corr_stocks = high_neg_corr_stocks.nsmallest(20)
                
                high_neg_corr_stocks = high_neg_corr_stocks[high_neg_corr_stocks < -0.6]
                graph_matrix_negative.loc[stock, high_neg_corr_stocks.index] = 1

        else:
            high_pos_corr_stocks = pos_corr_stocks[pos_corr_stocks > 0.72]
            if len(high_pos_corr_stocks) > 0:
                if len(high_pos_corr_stocks[high_pos_corr_stocks > 0.9]) > 20:
                    high_pos_corr_stocks = high_pos_corr_stocks[high_pos_corr_stocks > 0.9]
                else:
                    high_pos_corr_stocks = high_pos_corr_stocks.nlargest(20)
                high_pos_corr_stocks = high_pos_corr_stocks[high_pos_corr_stocks > 0.72]
                graph_matrix_positive.loc[stock, high_pos_corr_stocks.index] = 1
            
            high_neg_corr_stocks = neg_corr_stocks[neg_corr_stocks < -0.65]
            if len(high_neg_corr_stocks) > 0:
                if len(high_neg_corr_stocks[high_neg_corr_stocks < -0.8]) > 20:
                    high_neg_corr_stocks = high_neg_corr_stocks[high_neg_corr_stocks < -0.8]
                else:
                    high_neg_corr_stocks = high_neg_corr_stocks.nsmallest(20)
                high_neg_corr_stocks = high_neg_corr_stocks[high_neg_corr_stocks < -0.65]
                graph_matrix_negative.loc[stock, high_neg_corr_stocks.index] = 1
    
    np.fill_diagonal(graph_matrix_positive.values, 1)
    np.fill_diagonal(graph_matrix_negative.values, 1)

    graph_matrix_positive.to_csv(f'{output_folder}/graph_matrix_positive_{current_date}.csv')
    graph_matrix_negative.to_csv(f'{output_folder}/graph_matrix_negative_{current_date}.csv')

    graph_matrices_positive_dict[current_date] = graph_matrix_positive
    graph_matrices_negative_dict[current_date] = graph_matrix_negative

with open(f'{output_folder}/graph_matrices_positive_100.pkl', 'wb') as f:
    pickle.dump(graph_matrices_positive_dict, f)

with open(f'{output_folder}/graph_matrices_negative_100.pkl', 'wb') as f:
    pickle.dump(graph_matrices_negative_dict, f)

print("Graph matrices (positive and negative) have been saved in both CSV files and the pickle files.")


pkl_file_path = f'{output_folder}/graph_matrices_positive_100.pkl'
with open(pkl_file_path, 'rb') as f:
    correlation_matrices = pickle.load(f)

class ICLoss(nn.Module):
    def __init__(self, **kwargs):
        super(ICLoss, self).__init__()

    def forward(self, pred, label):
        pred = pred.squeeze()
        label = label.squeeze()
        vx = pred - torch.mean(pred)
        vy = label - torch.mean(label)
        cost = torch.sum(vx * vy) / (
            torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))
            )
        return -cost


merged_df['psudo_label_origin'] = merged_df['psudo_label'].apply(lambda x: 0 if pd.isna(x) else 1 if x == 1 else -1)
merged_df['psudo_label'] = merged_df['psudo_label'].apply(lambda x: 1 if pd.notna(x) else 0)
merged_df['calculated_avg'] = merged_df['calculated_avg'].apply(lambda x: x if pd.notna(x) else 0)
merged_df = merged_df.drop_duplicates(subset=['stock', 'datetime'], keep='first')

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


# model = AttentionGAT(in_channels=3, hidden_channels=36, out_channels=2).to(device)  
class AttentionGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1):
        super(AttentionGAT, self).__init__()
        
        # 使用GCN替换GAT
        # self.industry_conv1 = GCNConv(in_channels, hidden_channels)
        # self.industry_conv2 = GCNConv(hidden_channels, out_channels)
        # self.positive_corr_conv1 = GCNConv(in_channels, hidden_channels)
        # self.positive_corr_conv2 = GCNConv(hidden_channels, out_channels)
        
        # # 第一次hop后的attention权重
        # self.first_hop_attention = torch.nn.Parameter(torch.Tensor(2, hidden_channels))
        # torch.nn.init.xavier_uniform_(self.first_hop_attention)
        
        # # 最终的attention权重
        # self.attention_weight = torch.nn.Parameter(torch.Tensor(2, out_channels))
        # torch.nn.init.xavier_uniform_(self.attention_weight)
        
        # self.mlp = torch.nn.Linear(out_channels, 1)
        super(AttentionGAT, self).__init__()
        self.industry_conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.industry_conv2 = GATConv(hidden_channels * 1, out_channels, heads=heads)
        self.positive_corr_conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.positive_corr_conv2 = GATConv(hidden_channels * 1, out_channels, heads=heads)
        
        # 第一次hop后的attention权重
        self.first_hop_attention = torch.nn.Parameter(torch.Tensor(2, hidden_channels))
        torch.nn.init.xavier_uniform_(self.first_hop_attention)
        
        # 最终的attention权重
        self.attention_weight = torch.nn.Parameter(torch.Tensor(2, out_channels))
        torch.nn.init.xavier_uniform_(self.attention_weight)
        self.mlp = torch.nn.Linear(out_channels, 1)




    # def forward(self, industry_data, pos_corr_data):
    #     # 第一次hop
    #     x_industry = self.industry_conv1(industry_data.x, industry_data.edge_index)
    #     x_industry = F.relu(x_industry)
    #     x_pos_corr = self.positive_corr_conv1(pos_corr_data.x, pos_corr_data.edge_index)
    #     x_pos_corr = F.relu(x_pos_corr)

    #     x_industry = self.industry_conv2(x_industry, industry_data.edge_index)
    #     x_pos_corr = self.positive_corr_conv2(x_pos_corr, pos_corr_data.edge_index)

    #     x = x_industry
    #     x = self.mlp(x)
    #     return x


    def forward(self, industry_data, pos_corr_data):
        # 第一次hop
        x_industry = self.industry_conv1(industry_data.x, industry_data.edge_index)
        x_industry = F.relu(x_industry)
        
        x_pos_corr = self.positive_corr_conv1(pos_corr_data.x, pos_corr_data.edge_index)
        x_pos_corr = F.relu(x_pos_corr)
        
        # 第一次hop后融合特征
        alpha_first = F.softmax(torch.stack([
            torch.sum(self.first_hop_attention[0] * x_industry, dim=1),
            torch.sum(self.first_hop_attention[1] * x_pos_corr, dim=1)
        ], dim=0), dim=0)
        
        # 使用attention权重组合特征
        x_fused = alpha_first[0].unsqueeze(1) * x_industry + alpha_first[1].unsqueeze(1) * x_pos_corr
        
        # 使用融合后的特征进行第二次hop
        x_industry = self.industry_conv2(x_fused, industry_data.edge_index)
        x_pos_corr = self.positive_corr_conv2(x_fused, pos_corr_data.edge_index)
        
        # 最终的attention机制
        alpha_industry = self.attention_weight[0] * x_industry
        alpha_pos_corr = self.attention_weight[1] * x_pos_corr
        alpha_industry = torch.sum(alpha_industry, dim=1)
        alpha_pos_corr = torch.sum(alpha_pos_corr, dim=1)
        alpha = F.softmax(torch.stack([alpha_industry, alpha_pos_corr], dim=0), dim=0)
        x = alpha[0].unsqueeze(1) * x_industry + alpha[1].unsqueeze(1) * x_pos_corr
        x = self.mlp(x)
        return x

model = AttentionGAT(in_channels=2, hidden_channels=48, out_channels=2).to(device)


df_list = list(correlation_matrices.values())
stock_sort_list = df_list[-1].index.tolist()

def get_daily_graph_data(day):
    unique_dates = merged_df['datetime'].unique()
    day_date = unique_dates[day]
    df_merge_day = merged_df[merged_df['datetime'] == day_date]
    
    # df_merge_day['valid_value']=1
    x = df_merge_day[['valid_value', 'psudo_label', 'calculated_avg']].values

    # if x.isnull().any():
    #     print(x)
    y = df_merge_day['label_value'].values
    x = torch.tensor(x, dtype=torch.float).to(device)
    y = torch.tensor(y, dtype=torch.float).to(device)
    stock_day = df_merge_day['stock'].unique().tolist()
    stock_sort_day = stock_day
    df_day_matrix = df_list[day]
    n = len(stock_sort_day)
    adj_matrix_sorted = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if dict_sector[stock_sort_day[i]] == dict_sector[stock_sort_day[j]]:
                adj_matrix_sorted[i][j] = 1
    np.fill_diagonal(adj_matrix_sorted, 1)
    industry_adj = torch.tensor(adj_matrix_sorted, dtype=torch.float).to(device)
    industry_edge_index = dense_to_sparse(industry_adj)[0]
    df_matrix_sorted = df_day_matrix.loc[stock_sort_day, stock_sort_day].values
    np.fill_diagonal(df_matrix_sorted, 1)
    positive_corr_adj = torch.tensor(df_matrix_sorted, dtype=torch.float).to(device)
    positive_corr_edge_index = dense_to_sparse(positive_corr_adj)[0]
    valid_values = torch.tensor(df_merge_day['valid_value'].values, dtype=torch.float)
    calculated_avg = torch.tensor(df_merge_day['calculated_avg'].values, dtype=torch.float)
    updated_valid_values = torch.where(calculated_avg != 0, calculated_avg, valid_values)
    label_values = torch.tensor(df_merge_day['label_value'].values, dtype=torch.float)

    correct = (valid_values.sign() == label_values.sign()).sum().item()
    correct_modify = (updated_valid_values.sign() == label_values.sign()).sum().item()
    total = valid_values.size(0)
    accuracy = correct / total
    acc_mod = correct_modify / total
    return x, y, industry_edge_index, positive_corr_edge_index, accuracy, acc_mod,day_date


optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
preserve_i = []

def prepare_data(df_list):
    x_list, y_list, industry_edge_list, pos_corr_edge_list, acc_base_list, acc_mod_list = [], [], [], [], [], []

    day_date_list = []
    for i in range(len(df_list)):
        x, y, industry_edge_index, pos_corr_edge_index, acc_base, acc_mod, day_date = get_daily_graph_data(i)
        num_i = len(x)
        if True:
            day_date_list.extend([day_date] * num_i)
            preserve_i.append(i)
            x_list.append(x)
            y_list.append(y)
            industry_edge_list.append(industry_edge_index)
            pos_corr_edge_list.append(pos_corr_edge_index)
            acc_base_list.append(acc_base)
            acc_mod_list.append(acc_mod)
    return x_list, y_list, industry_edge_list, pos_corr_edge_list, acc_base_list, acc_mod_list, day_date_list

x_list, y_list, industry_edge_list, pos_corr_edge_list, acc_base_list, acc_mod_list, day_date_list = prepare_data(df_list)


day_date_series = pd.to_datetime(day_date_list)

split_index_21 = (day_date_series <= '2021-01-01').sum()  
split_index_22 = (day_date_series <= '2022-01-04').sum()  

total_samples = len(day_date_list)


# 计算各时间段样本比例
total_samples = len(day_date_list)
proportion_2021_before = split_index_21 / total_samples
proportion_2022_before = split_index_22 / total_samples

print(f'Before 2021 proportion: {proportion_2021_before:.4f}')
print(f'Before 2022 proportion: {proportion_2022_before:.4f}')


split_index_21 = int(proportion_2021_before * len(x_list)) 
split_index_22 = int(proportion_2022_before * len(x_list)) 
train_idx = list(range(split_index_21))
test_idx = list(range(split_index_22, len(x_list)))
valid_idx = list(range(split_index_21, split_index_22))


train_data = [(x_list[i], y_list[i], industry_edge_list[i], pos_corr_edge_list[i], acc_base_list[i], acc_mod_list[i]) for i in train_idx]
test_data = [(x_list[i], y_list[i], industry_edge_list[i], pos_corr_edge_list[i], acc_base_list[i], acc_mod_list[i]) for i in test_idx]
valid_data = [
    (x_list[i], y_list[i], industry_edge_list[i], pos_corr_edge_list[i], acc_base_list[i], acc_mod_list[i])
    for i in valid_idx
]
ic_loss = ICLoss()

def train(epoch):
    model.train()
    total_loss = 0
    for x, y, industry_edge_index, pos_corr_edge_index, acc_base, acc_mod in train_data:
        x = x.to(device)
        y = y.to(device)
        industry_edge_index = industry_edge_index.to(device)
        pos_corr_edge_index = pos_corr_edge_index.to(device)
        industry_data = Data(x=x, edge_index=industry_edge_index)
        pos_corr_data = Data(x=x, edge_index=pos_corr_edge_index)
        optimizer.zero_grad()
        out = model(industry_data, pos_corr_data)
        
        
        condition = x[:, -1] != 0
        # out.squeeze()
        # out = torch.where(condition, x[:, 2], out.squeeze())
        #       
        # loss=ic_loss(out, y)
        loss = F.l1_loss(out, y)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    return total_loss / len(train_data)

def dump_predict_results(pred_series, y_series,long_short_r):
        csv_dump_path = 'tmp/dump_result_nastmp'
        os.makedirs(csv_dump_path, exist_ok=True)
        pred_df = pred_series.unstack()
        y_df = y_series.unstack()
        pred_df.to_csv(csv_dump_path + "/pred.csv")
        y_df.to_csv(csv_dump_path + "/label.csv")
        long_short_r.to_csv(csv_dump_path + "/long_short.csv")


def compute_validation_metrics(pred_series, y_series, psudo_series, pred_bce_series,quantile=0.1):
    stock_list_nas=stock_list[:98]
    mask = pred_series.index.get_level_values('instrument').isin(stock_list_nas)
    pred_series = pred_series[mask]
    y_series = y_series[mask]
    assert pred_series.index.equals(y_series.index)
    y_series1 = y_series.apply(lambda x: 1 if x >= 0 else 0)
    pred_series1 = pred_series.apply(lambda x: 1 if x >= 0 else 0)
    comparison = y_series1 == pred_series1
    acc_ic = comparison.mean()
    
    try:
        ic1, ric = calc_ic(pred_series, y_series)
        icir = ic1.mean() / ic1.std() if ic1.std() != 0 else float('inf')
        # long_short_r, long_avg_r = calc_long_short_return(pred_series, y_series)
        v_rmse = rmse(pred_series, y_series)
        v_ic = ic(pred_series, y_series)
        long_short_r, long_avg_r, long_short_ann_return, long_short_ann_sharpe, long_avg_ann_return, long_avg_ann_sharpe = calc_long_short_annual_return(pred_series, y_series,quantile=quantile)
        long_short_r, long_avg_r = calc_long_short_return(pred_series, y_series,quantile=quantile)
        dump_predict_results(pred_series, y_series,long_short_r)
        if pred_bce_series is not None:
            pred_bce_label = pred_bce_series.apply(lambda x: x.argmax())
            comparison_bce = y_series1 == pred_bce_label
            acc_bce = comparison_bce.mean()
            valid_metric_dict = {
                "acc_bce": acc_bce,
                "acc_ic": acc_ic,
                "valid_rmse": v_rmse,
                "valid_ic": v_ic,
                "valid_ric": ric.mean(),
                "ICIR": icir,
                "Long-Short Mean Return": long_short_r.mean() * 252,
                "Long-Short Mean Sharpe": long_short_r.mean() / long_short_r.std() * 252 ** 0.5,
                # "Long-Avg Ann Return": long_avg_r.mean() * 252,
                # "Long-Avg Ann Sharpe": long_avg_r.mean() / long_avg_r.std() * 252 ** 0.5
                "Long-Short Ann Return": long_short_ann_return,
                "Long-Short Ann Sharpe": long_short_ann_sharpe,
                "Long-Avg Ann Return": long_avg_ann_return,
                "Long-Avg Ann Sharpe": long_avg_ann_sharpe
            }
        else:
            valid_metric_dict = {
                "acc_ic": acc_ic,
                "valid_rmse": v_rmse,
                "valid_ic": v_ic,
                "valid_ric": ric.mean(),
                "ICIR": icir,
                "Long-Short Mean Return": long_short_r.mean() * 252,
                "Long-Short Mean Sharpe": long_short_r.mean() / long_short_r.std() * 252 ** 0.5,
                "Long-Short Ann Return": long_short_ann_return,
                "Long-Short Ann Sharpe": long_short_ann_sharpe,
                "Long-Avg Ann Return": long_avg_ann_return,
                "Long-Avg Ann Sharpe": long_avg_ann_sharpe,
            }
    except BaseException as e:
        print(e)
    return valid_metric_dict
import time

def evaluate(valid_data,valid_idx,quantile= 0.1):
    model.eval()
    num_1 = 0
    num_2 = 0
    num_3 = 0
    acc_1 = 0
    acc_2 = 0
    acc_3=0
    all_pred = []
    all_pred_zero=[]
    all_true_zero_y=[]
    all_true = []
    all_dates = []
    all_dates_zero=[]
    all_stocks_zero=[]
    all_stocks = []
    all_origin = []
    time1=time.time()
    with torch.no_grad():
        for i, (x, y, industry_edge_index, pos_corr_edge_index, acc_base, acc_mod) in enumerate(valid_data):
            if test_idx[i] in preserve_i:
                x = x.to(device)
                y = y.to(device)
                industry_edge_index = industry_edge_index.to(device)
                pos_corr_edge_index = pos_corr_edge_index.to(device)
                industry_data = Data(x=x, edge_index=industry_edge_index)
                pos_corr_data = Data(x=x, edge_index=pos_corr_edge_index)
                out = model(industry_data, pos_corr_data)
                
                condition = x[:, -1] != 0
                # out = torch.where(condition, x[:, 2], out.squeeze())
                out = out.squeeze()
                
                x = x[:,-1]
                # 1. x != 0 的位置
                indices_x_nonzero = (x != 0)
                sign_x = torch.sign(x[indices_x_nonzero])
                sign_y_x = torch.sign(y[indices_x_nonzero])
                correct_x = (sign_x == sign_y_x).sum().item()
                total_x = indices_x_nonzero.sum().item()
                accuracy_x_nonzero = correct_x / total_x if total_x > 0 else None

                # 2. output_two_hop != 0 的位置
                indices_output_two_hop_nonzero = (out != 0)
                sign_output_two_hop = torch.sign(out[indices_output_two_hop_nonzero])
                sign_y_output = torch.sign(y[indices_output_two_hop_nonzero])
                correct_output = (sign_output_two_hop == sign_y_output).sum().item()
                total_output = indices_output_two_hop_nonzero.sum().item()
                accuracy_output_two_hop_nonzero = correct_output / total_output if total_output > 0 else None

                # 3. x == 0 且 output_two_hop != 0 的位置
                indices_x_zero_output_nonzero = (x == 0) & (out != 0)

                all_pred_zero.append(out[indices_x_zero_output_nonzero].cpu().numpy())
                all_true_zero_y.append(y[indices_x_zero_output_nonzero].cpu().numpy())

                sign_output_x_zero = torch.sign(out[indices_x_zero_output_nonzero])
                sign_y_x_zero = torch.sign(y[indices_x_zero_output_nonzero])
                correct_x_zero_output = (sign_output_x_zero == sign_y_x_zero).sum().item()
                total_x_zero_output = indices_x_zero_output_nonzero.sum().item()
                accuracy_x_zero_output_nonzero = correct_x_zero_output / total_x_zero_output if total_x_zero_output > 0 else None
                
                all_pred.append(out.cpu().numpy())
                all_true.append(y.cpu().numpy())
                unique_dates = merged_df['datetime'].unique()
                unique_dates = unique_dates[preserve_i]
                day_date = unique_dates[valid_idx[i]]
                df_merge_day = merged_df[merged_df['datetime'] == day_date]
                all_dates.append(df_merge_day['datetime'].values)
                all_dates_zero.append(df_merge_day['datetime'].values[indices_x_zero_output_nonzero.cpu()])
                all_stocks.append(df_merge_day['stock'].values)
                all_stocks_zero.append(df_merge_day['stock'].values[indices_x_zero_output_nonzero.cpu()])
                valid_origin = df_merge_day['valid_value'].values
                all_origin.append(valid_origin)


                try:
                    acc_1 += accuracy_x_nonzero
                    num_1+=1
                except BaseException as e:
                    pass    
                
                try:
                    acc_2+=accuracy_output_two_hop_nonzero
                    num_2+=1
                except BaseException as e:
                    pass      
                    
                try:
                    acc_3+=accuracy_x_zero_output_nonzero
                    num_3+=1
                except BaseException as e:
                    pass    
                
    time2=time.time()
    print(f'Time taken for evaluation: {time2 - time1:.4f} seconds')
    all_pred_zero = np.concatenate(all_pred_zero) 
    all_true_zero_y = np.concatenate(all_true_zero_y) 
    all_stocks_zero = np.concatenate(all_stocks_zero)         
    all_pred = np.concatenate(all_pred)
    all_true = np.concatenate(all_true)
    all_dates = np.concatenate(all_dates)
    all_dates_zero = np.concatenate(all_dates_zero)
    all_stocks = np.concatenate(all_stocks)
    all_origin = np.concatenate(all_origin)
    
    
    pred_series_zero = pd.Series(all_pred_zero, index=pd.MultiIndex.from_arrays([all_dates_zero, all_stocks_zero], names=["datetime", "instrument"]))
    y_series_zero = pd.Series(all_true_zero_y, index=pd.MultiIndex.from_arrays([all_dates_zero, all_stocks_zero], names=["datetime", "instrument"]))
    
    
    pred_series = pd.Series(all_pred, index=pd.MultiIndex.from_arrays([all_dates, all_stocks], names=["datetime", "instrument"]))
    y_series = pd.Series(all_true, index=pd.MultiIndex.from_arrays([all_dates, all_stocks], names=["datetime", "instrument"]))
    
    
    pred_series_origin = pd.Series(all_origin, index=pd.MultiIndex.from_arrays([all_dates, all_stocks], names=["datetime", "instrument"]))
    valid_metrics_zero = compute_validation_metrics(pred_series_zero, y_series_zero, None, None,quantile=quantile)
    
    valid_metrics = compute_validation_metrics(pred_series, y_series, None, None,quantile=quantile)
    valid_metrics_origin = compute_validation_metrics(pred_series_origin, y_series, None, None,quantile=quantile)
    return acc_1 / num_1, acc_2 / num_2, acc_3 / num_3, valid_metrics, valid_metrics_origin,valid_metrics_zero
    # return valid_metrics, valid_metrics_origin
valid_ic_best = -0.1
for epoch in range(50):
    if epoch == 30:
        print('epoch is:',epoch)
    loss = train(epoch)
    if (epoch >= 0) and (epoch % 5) == 0:
        accuracy, acc_base, acc_mod, valid_metric_out, valid_metric_origin,valid_metric_zero = evaluate(valid_data=test_data,valid_idx=test_idx)
        valid_ic = valid_metric_out['valid_ic']
        if valid_ic >= valid_ic_best:
            valid_ic_best = valid_ic
            print('Best epoch: ',epoch)
            torch.save(model.state_dict(), 'best_model_nasdaqonly.pth')
        print(f'Validation Epoch {epoch}, Loss: {loss:.4f}, Accuracy1: {accuracy:.4f}')
        print('Output metrics: ', valid_metric_out)
        print('Zero metrics: ', valid_metric_zero)

model.load_state_dict(torch.load('best_model_nasdaqonly.pth'))
model.eval()  
accuracy, acc_base, acc_mod, valid_metric_out, valid_metric_origin,valid_metric_zero = evaluate(valid_data=valid_data,valid_idx=valid_idx,quantile= 0.1)
print(f'Test information, Accuracy1: {accuracy:.4f}')
print('Output metrics_0.1: ', valid_metric_out)
print('Zero metrics_0.1: ', valid_metric_zero)

accuracy, acc_base, acc_mod, valid_metric_out, valid_metric_origin,valid_metric_zero = evaluate(valid_data=test_data,valid_idx=test_idx,quantile= 0.2)
print(f'Test information, Accuracy1: {accuracy:.4f}')
print('Output metrics_0.2: ', valid_metric_out)
print('Zero metrics_0.2: ', valid_metric_zero)