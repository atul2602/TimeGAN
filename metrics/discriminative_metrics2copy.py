import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score
from utils import train_test_divide, extract_time, batch_generator


def discriminative_score_metrics(ori_data, generated_data):
    """Use post-hoc RNN to classify original data and synthetic data
    
    Args:
        - ori_data: original data
        - generated_data: generated synthetic data
    
    Returns:
        - discriminative_score: np.abs(classification accuracy - 0.5)
    """
    # Initialization
    torch.manual_seed(42)

    # Basic Parameters
    no, seq_len, dim = np.asarray(ori_data).shape

    # Set maximum sequence length and each sequence length
    ori_time, ori_max_seq_len = extract_time(ori_data)
    generated_time, generated_max_seq_len = extract_time(ori_data)
    max_seq_len = max([ori_max_seq_len, generated_max_seq_len])

    ## Build a post-hoc RNN discriminator network
    # Network parameters
    hidden_dim = int(dim / 2)
    iterations = 2000
    batch_size = 128

    # Discriminator model
    class Discriminator(nn.Module):
        def __init__(self, input_dim, hidden_dim):
            super(Discriminator, self).__init__()
            self.rnn = nn.GRU(input_dim, hidden_dim)
            self.fc = nn.Linear(hidden_dim, 1)

        def forward(self, x, t):
            _, h_n = self.rnn(x)
            y_hat_logit = self.fc(h_n.squeeze())
            y_hat = torch.sigmoid(y_hat_logit)
            return y_hat_logit, y_hat

    discriminator = Discriminator(dim, hidden_dim)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(discriminator.parameters())

    ## Train the discriminator
    # Train/test division for both original and generated data
    train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat = \
        train_test_divide(ori_data, generated_data, ori_time, generated_time)

    # Training loop
    for itt in range(iterations):
        # Batch setting
        X_mb, T_mb = batch_generator(train_x, train_t, batch_size)
        X_hat_mb, T_hat_mb = batch_generator(train_x_hat, train_t_hat, batch_size)

        # Train discriminator
        optimizer.zero_grad()
        y_logit_real, _ = discriminator(X_mb, T_mb)
        y_logit_fake, _ = discriminator(X_hat_mb, T_hat_mb)
        d_loss_real = criterion(y_logit_real, torch.ones_like(y_logit_real))
        d_loss_fake = criterion(y_logit_fake, torch.zeros_like(y_logit_fake))
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer.step()

    ## Test the performance on the testing set
    with torch.no_grad():
        y_logit_real, _ = discriminator(test_x, test_t)
        y_logit_fake, _ = discriminator(test_x_hat, test_t_hat)
        y_pred_real = (torch.sigmoid(y_logit_real) > 0.5).cpu().numpy().flatten()
        y_pred_fake = (torch.sigmoid(y_logit_fake) > 0.5).cpu().numpy().flatten()

    y_pred_final = np.concatenate((y_pred_real, y_pred_fake), axis=0)
    y_label_final = np.concatenate((np.ones_like(y_pred_real), np.zeros_like(y_pred_fake)), axis=0)

    # Compute the accuracy
    acc = accuracy_score(y_label_final, y_pred_final)
    discriminative_score = np.abs(0.5 - acc)

    return discriminative_score
