import numpy as np
import torch

def train_test_divide (data_x, data_x_hat, data_t, data_t_hat, train_rate = 0.8):
  """Divide train and test data for both original and synthetic data.
  
  Args:
    - data_x: original data
    - data_x_hat: generated data
    - data_t: original time
    - data_t_hat: generated time
    - train_rate: ratio of training data from the original data
  """
  # Divide train/test index (original data)
  no = len(data_x)
  idx = np.random.permutation(no)
  train_idx = idx[:int(no*train_rate)]
  test_idx = idx[int(no*train_rate):]
    
  train_x = [data_x[i] for i in train_idx]
  test_x = [data_x[i] for i in test_idx]
  train_t = [data_t[i] for i in train_idx]
  test_t = [data_t[i] for i in test_idx]      
    
  # Divide train/test index (synthetic data)
  no = len(data_x_hat)
  idx = np.random.permutation(no)
  train_idx = idx[:int(no*train_rate)]
  test_idx = idx[int(no*train_rate):]
  
  train_x_hat = [data_x_hat[i] for i in train_idx]
  test_x_hat = [data_x_hat[i] for i in test_idx]
  train_t_hat = [data_t_hat[i] for i in train_idx]
  test_t_hat = [data_t_hat[i] for i in test_idx]

  print(" Hooray: train_x size: ", np.shape(np.array(train_x)))
  print(" Hooray: test_x size: ", np.shape(np.array(test_x)))
  return train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat


def extract_time(data):
    time = []
    max_seq_len = 0
    for i in range(len(data)):
        max_seq_len = max(max_seq_len, data[i].shape[0])
        time.append(data[i].shape[0])
    
    return time, max_seq_len


class RNNCellWrapper(torch.nn.Module):
    def __init__(self, module_name, hidden_dim):
        super(RNNCellWrapper, self).__init__()
        assert module_name in ['gru', 'lstm', 'lstmLN']
        self.module_name = module_name
        self.hidden_dim = hidden_dim
        
        if self.module_name == 'gru':
            self.rnn_cell = torch.nn.GRUCell(hidden_dim, hidden_dim)
        elif self.module_name == 'lstm':
            self.rnn_cell = torch.nn.LSTMCell(hidden_dim, hidden_dim)
        elif self.module_name == 'lstmLN':
            self.rnn_cell = torch.nn.LayerNormLSTMCell(hidden_dim, hidden_dim)
        
    def forward(self, x, h):
        if self.module_name == 'lstmLN':
            h, c = h
            h, c = self.rnn_cell(x, (h, c))
            return h, (h, c)
        else:
            return self.rnn_cell(x, h)


def random_generator(batch_size, z_dim, T_mb, max_seq_len):
    Z_mb = []
    for i in range(batch_size):
        temp = np.zeros((max_seq_len, z_dim))
        temp_Z = np.random.uniform(0., 1, (T_mb[i], z_dim))
        temp[:T_mb[i], :] = temp_Z
        Z_mb.append(temp_Z)
    return Z_mb


def batch_generator(data, time, batch_size):
    no = len(data)
    idx = np.random.permutation(no)
    train_idx = idx[:batch_size]
    
    X_mb = [data[i] for i in train_idx]
    T_mb = [time[i] for i in train_idx]
  
    return X_mb, T_mb
