import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import datetime
import torch.nn.functional as F
import csv
import copy

from sklearn.metrics import mean_squared_error, mean_absolute_error


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    print(f"current GPU: {gpu_name}")
    
class PhysicsInformedNN(nn.Module):
    def __init__(self, input_size=300, hidden_size=512, output_size=100, mapping_function='psi1'):
        super().__init__()
        self.utau_r = nn.Parameter(torch.tensor(0.7))
        self.utau_d = nn.Parameter(torch.tensor(2.0))
        self.tau_r_min = 0.1
        self.tau_r_max = 1.4
        self.tau_d_min = 1.5
        self.tau_d_max = 6.0
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.cnn = nn.Sequential(
            nn.Conv1d(3, hidden_size // 8 , kernel_size=3, stride=1, padding=1),  
            nn.Tanh(),
            nn.BatchNorm1d(hidden_size // 8),
            nn.Conv1d(hidden_size // 8, hidden_size//4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(hidden_size // 4),
            nn.Conv1d(hidden_size // 4, hidden_size//2, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm1d(hidden_size // 2),
        )
        self.lstm = nn.LSTM(
            input_size=hidden_size//2,
            hidden_size=hidden_size,  
            num_layers=2,     
            bidirectional=False,
            batch_first=True, 
            dropout=0.1
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 1 ),
        )
        self.register_buffer('dy_kernel', 
            (torch.tensor([-1., 0., 1.]) / 2).view(1, 1, 3),
            persistent=False  
        )
        self.register_buffer('d2y_kernel', 
            torch.tensor([1., -2., 1.]).view(1, 1, 3),
            persistent=False)
        
        self._init_mapping_function(mapping_function)

    def _init_mapping_function(self, mapping_function):
        self.mapping_function_type = mapping_function
        base_params = {
            'a': nn.Parameter(torch.randn(1) * 0.1 + 0.5),  
            'c': nn.Parameter(torch.zeros(1))       
        }

        test_functions = ['psi1', 'psi2', 'psi3', 'e', 'ln', 'tan']
        
        self.mapping_params = nn.ParameterDict(base_params)

    def tau_r(self):
        return self.tau_r_min + (self.tau_r_max - self.tau_r_min) * torch.sigmoid(self.utau_r)
        

    def tau_d(self):
        return self.tau_d_min + (self.tau_d_max - self.tau_d_min) * torch.sigmoid(self.utau_d)
        
    
    def predict_psi(self, u_pred):
        u_pred_normalized = u_pred
        EPS = 1e-6  
        a = self.mapping_params['a'] + EPS
        raw_c = self.mapping_params['c']
        c = F.softplus(raw_c) 

        if self.mapping_function_type == 'psi1':
            psi = (u_pred - c) / a
            
        elif self.mapping_function_type == 'psi2':
            input_ = (u_pred - c)/a
            input_ = torch.clamp(input_, min=-1e3, max=1e3)  
            psi = torch.pow(torch.abs(input_) + EPS, 1.0/2) * torch.sign(input_)
        
        if self.mapping_function_type == 'psi3':
            input_ = (u_pred - c)/a
            input_ = torch.clamp(input_, min=-1e3, max=1e3)  
            psi = torch.pow(torch.abs(input_) + EPS, 1.0/3) * torch.sign(input_)
            
        elif self.mapping_function_type == 'e':
            input_ = (u_pred - c)/a
            input_ = torch.clamp(input_, min=EPS, max=1e6)  
            psi = torch.log(input_)
        
        elif self.mapping_function_type == 'ln': 
            psi = torch.exp((u_pred_normalized-c)/a) - 1
        
        elif self.mapping_function_type == 'tan':
            psi = 20 / torch.pi * torch.atan((u_pred_normalized-c)/a)

        return psi
    
    def forward(self, y):
        if y.size(-1) != 1:
            y = y.unsqueeze(-1) 

        B, T, _ = y.shape

        y_padded = F.pad(y.permute(0, 2, 1), (1, 1), mode='reflect')  
        
        dy = torch.conv1d(y_padded, self.dy_kernel, padding=0) 
        dy = dy.permute(0, 2, 1)  

        d2y = torch.conv1d(y_padded, self.d2y_kernel, padding=0) 
        d2y = d2y.permute(0, 2, 1)  

        features = torch.cat([y, dy, d2y], dim=-1)  

        cnn_input = features.permute(0, 2, 1)  

        cnn_out = self.cnn(cnn_input)  

        lstm_input = cnn_out.permute(0, 2, 1) 
        lstm_out, _ = self.lstm(lstm_input) 
        
        u_net = self.fc(lstm_out)  

        u_eqn = self.tau_r() * self.tau_d() * d2y + (self.tau_r() + self.tau_d()) * dy + y

        psi_pred = self.predict_psi(u_net)
        
        raw_c = self.mapping_params['c']
        c = F.softplus(raw_c)
        self.c_dynamic = c.detach()

        return psi_pred, u_net, u_eqn
   

class DynamicLoss(nn.Module):
    def __init__(self, phy_weight=1.0, psi_weight=1.0):
        super().__init__()
        self.phy_weight = phy_weight
        self.psi_weight = psi_weight

    def forward(self, psi_pred, psi_true, u_net, u_eqn):
        
        psi_pred = psi_pred.squeeze(-1)       
        psi_true = psi_true.view_as(psi_pred) 
        u_net = u_net.squeeze(-1) 
        u_eqn = u_eqn.view_as(psi_pred)
        
        phy_loss = torch.mean((u_net - u_eqn) ** 2)
        
        psi_loss = F.mse_loss(psi_pred, psi_true)
        
        total_loss = (
            self.phy_weight * phy_loss 
            + self.psi_weight * psi_loss
        )
        
        with torch.no_grad():
            
            psi_true_mean = torch.mean(psi_true)
            ss_res = torch.sum((psi_true - psi_pred) ** 2)      
            ss_total = torch.sum((psi_true - psi_true_mean) ** 2)                
            test_mae = torch.abs(psi_pred - psi_true).mean()
            
        return (
            total_loss,
            phy_loss.item(),
            psi_loss.item(),
            test_mae.item()
        )


def load_and_preprocess_data(phys_data_file, annotations_data_file, random_seed, segment_length=50):
    
    phys_df = pd.read_csv(phys_data_file)
    annotations_df = pd.read_csv(annotations_data_file)
    
    video_labels = annotations_df['video'].values
    
    y_data = phys_df['scr'].values
    psi_data = annotations_df['arousal'].values
    
    y_segments = []
    psi_segments = []
    video_ids = []

    for video_id in np.unique(video_labels):
        mask = video_labels == video_id
        y_video = y_data[mask]
        psi_video = psi_data[mask]
        
        num_segments = len(y_video) // segment_length
        for i in range(num_segments):
            start = i * segment_length
            end = start + segment_length
            y_segments.append(y_video[start:end])
            psi_segments.append(psi_video[start:end])
            video_ids.append(video_id)  

    indices = np.arange(len(y_segments))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    
    y_shuffled = [y_segments[i] for i in indices]
    psi_shuffled = [psi_segments[i] for i in indices]
    video_ids_shuffled = [video_ids[i] for i in indices]

    total = len(y_shuffled)
    train_end = int(total * 0.7)
    val_end = train_end + int(total * 0.15)

    y_train = y_shuffled[:train_end]
    psi_train = psi_shuffled[:train_end]
    
    y_val = y_shuffled[train_end:val_end]
    psi_val = psi_shuffled[train_end:val_end]
    
    y_test = y_shuffled[val_end:]
    psi_test = psi_shuffled[val_end:]

    
    def fit_scaler(data):
        concat_data = np.concatenate(data)
        scaler = MinMaxScaler(feature_range=(-1, 1)).fit(concat_data.reshape(-1,1))
        return scaler

    
    scaler_y = fit_scaler(y_train)
    scaler_psi = fit_scaler(psi_train)

    
    def safe_transform(data, scaler):
        return [scaler.transform(seg.reshape(-1,1)).flatten() for seg in data]

    y_train = safe_transform(y_train, scaler_y)
    y_val = safe_transform(y_val, scaler_y)
    y_test = safe_transform(y_test, scaler_y)
    
    psi_train = safe_transform(psi_train, scaler_psi)
    psi_val = safe_transform(psi_val, scaler_psi)
    psi_test = safe_transform(psi_test, scaler_psi)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def to_tensor(data):
        stacked = np.stack(data)[:, :, np.newaxis] 
        return torch.from_numpy(stacked).float().to(device)

    return (
        to_tensor(y_train), to_tensor(psi_train),
        to_tensor(y_val), to_tensor(psi_val),
        to_tensor(y_test), to_tensor(psi_test)
    )

def plot_results(psi_true, psi_pred, save_dir, datetime_str, dataset_type='train', figsize=(12, 5), dpi=300, loss=None):

    min_length = min(len(psi_true), len(psi_pred))
    psi_true = psi_true[:min_length]
    psi_pred = psi_pred[:min_length]
    try:
        if len(psi_true) == 0:
            raise ValueError("psi_true null")
        if len(psi_pred) == 0:
            raise ValueError("psi_pred null")
            
        mse = mean_squared_error(psi_true, psi_pred)
        mae = mean_absolute_error(psi_true, psi_pred)
        
    except Exception as e:
        print(f"false: {str(e)}")
        mse = mae = np.nan
    x = np.arange(min_length)
    os.makedirs(save_dir, exist_ok=True)
    
    x = np.arange(len(psi_true))

    plt.figure(figsize=figsize)
    plt.plot(x, psi_pred, label='Predicted', linestyle='--', linewidth=2, color='orange')
    plt.plot(x, psi_true, label='True', linewidth=2)
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('Arousal', fontsize=12)
    plt.title(f'{dataset_type} Results - MSE: {mse:.4f} - MAE: {mae:.4f}', fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f'{datetime_str}_{dataset_type}_results.png')
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()
    plt.close()

def valid(model, data_loader, loss_fn, device):
    
    model.eval()
    total_loss = 0.0
    total_phy = 0.0
    total_mse = 0.0
    total_mae = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for y_batch, psi_batch in data_loader:
            
            y_batch = y_batch.to(device)
            psi_batch = psi_batch.to(device)
            psi_pred, u_net, u_eqn = model(y_batch)
            loss, phy_loss, psi_loss, mae = loss_fn(psi_pred, psi_batch, u_net, u_eqn)
            total_loss += loss.item()
            total_phy += phy_loss
            total_mse += psi_loss
            total_mae += mae
            num_batches += 1

    
    metrics = {
        'loss': total_loss / num_batches if num_batches > 0 else 0,
        'phy': total_phy / num_batches if num_batches > 0 else 0,
        'mse': total_mse / num_batches if num_batches > 0 else 0,
        'mae': total_mae / num_batches if num_batches > 0 else 0,
        'num_samples': num_batches * data_loader.batch_size
    }
    
    return metrics

def test(model, data_loader, loss_fn, device):

    model.eval()
    total_loss = 0.0
    total_phy = 0.0
    total_mse = 0.0
    total_mae = 0.0

    num_batches = 0
    
    with torch.no_grad():
        for y_batch, psi_batch in data_loader:
            
            y_batch = y_batch.to(device)
            psi_batch = psi_batch.to(device)
            psi_pred, u_net, u_eqn = model(y_batch)
            loss, phy_loss, psi_loss, mae = loss_fn(psi_pred, psi_batch, u_net, u_eqn)
            total_loss += loss.item()
            total_phy += phy_loss
            total_mse += psi_loss
            total_mae += mae
            num_batches += 1

    metrics = {
        'loss': total_loss / num_batches if num_batches > 0 else 0,
        'phy': total_phy / num_batches if num_batches > 0 else 0,
        'mse': total_mse / num_batches if num_batches > 0 else 0,
        'mae': total_mae / num_batches if num_batches > 0 else 0,
        'num_samples': num_batches * data_loader.batch_size
    }
    
    return metrics

def plot_internal_states(y_true, u_net, u_eqn, sample_index=0, time_steps=None):


    u_net_np = u_net.cpu().numpy()[sample_index, :, 0]
    u_eqn_np = u_eqn.cpu().numpy()[sample_index, :, 0]
    plt.figure(figsize=(12, 5))
    plt.plot(u_net_np, label='u_net', color='royalblue', linewidth=2)
    plt.plot(u_eqn_np, label='u_eqn', color='crimson', linestyle='--', linewidth=2)
    plt.title('u ')
    plt.xlabel('t')
    plt.ylabel('u ')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
