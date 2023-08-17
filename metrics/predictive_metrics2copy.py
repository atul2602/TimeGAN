import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_absolute_error
from utils import extract_time  # You'll need to define the 'extract_time' function

class Predictor(nn.Module):
    def __init__(self, input_size, hidden_dim):
        super(Predictor, self).__init__()
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x, t):
        _, p_last_states = self.rnn(x)
        y_hat_logit = self.fc(p_last_states)
        y_hat = torch.sigmoid(y_hat_logit)
        return y_hat

def predictive_score_metrics(ori_data, generated_data):
    # Basic Parameters
    no, seq_len, dim = np.asarray(ori_data).shape
    
    # Set maximum sequence length and each sequence length
    ori_time, ori_max_seq_len = extract_time(ori_data)
    generated_time, generated_max_seq_len = extract_time(ori_data)
    max_seq_len = max([ori_max_seq_len, generated_max_seq_len])  
    
    # Network parameters
    hidden_dim = dim // 2
    iterations = 5000
    batch_size = 128
    
    predictor = Predictor(input_size=dim - 1, hidden_dim=hidden_dim)
    p_optimizer = optim.Adam(predictor.parameters())
    loss_fn = nn.L1Loss()
    
    # Training using Synthetic dataset
    for itt in range(iterations):
        idx = np.random.permutation(len(generated_data))
        train_idx = idx[:batch_size]
        
        X_mb = torch.tensor([generated_data[i][:-1, :(dim - 1)] for i in train_idx], dtype=torch.float32)
        T_mb = torch.tensor([generated_time[i] - 1 for i in train_idx], dtype=torch.int64)
        Y_mb = torch.tensor([generated_data[i][1:, (dim - 1)] for i in train_idx], dtype=torch.float32)
        
        p_optimizer.zero_grad()
        y_pred = predictor(X_mb, T_mb)
        p_loss = loss_fn(y_pred, Y_mb)
        p_loss.backward()
        p_optimizer.step()
    
    # Test the trained model on the original data
    idx = np.random.permutation(len(ori_data))
    train_idx = idx[:no]
    
    X_mb = torch.tensor([ori_data[i][:-1, :(dim - 1)] for i in train_idx], dtype=torch.float32)
    T_mb = torch.tensor([ori_time[i] - 1 for i in train_idx], dtype=torch.int64)
    Y_mb = torch.tensor([np.reshape(ori_data[i][1:,(dim-1)], [len(ori_data[i][1:,(dim-1)]),1]) for i in train_idx], dtype = torch.float32)
    
    pred_Y_curr = predictor(X_mb, T_mb)
    print(Y_mb[0].size())
    print(pred_Y_curr.size())
    
    # Compute the performance in terms of MAE
    MAE_temp = 0
    for i in range(no):
        MAE_temp += mean_absolute_error(Y_mb[i].detach().numpy(), pred_Y_curr[i].detach().numpy())
    
    predictive_score = MAE_temp / no
    
    return predictive_score
