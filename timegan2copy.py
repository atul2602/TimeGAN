import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils import extract_time, rnn_cell, random_generator, batch_generator


def timegan(ori_data, parameters):
    """TimeGAN function.
    
    Use original data as training set to generate synthetic data (time-series)
    
    Args:
        - ori_data: original time-series data
        - parameters: TimeGAN network parameters
        
    Returns:
        - generated_data: generated time-series data
    """
    # Basic Parameters
    no, seq_len, dim = np.asarray(ori_data).shape

    # Maximum sequence length and each sequence length
    ori_time, max_seq_len = extract_time(ori_data)

    def MinMaxScaler(data):
        """Min-Max Normalizer.
        
        Args:
            - data: raw data
            
        Returns:
            - norm_data: normalized data
            - min_val: minimum values (for renormalization)
            - max_val: maximum values (for renormalization)
        """
        min_val = np.min(np.min(data, axis=0), axis=0)
        data = data - min_val

        max_val = np.max(np.max(data, axis=0), axis=0)
        norm_data = data / (max_val + 1e-7)

        return norm_data, min_val, max_val

    # Normalization
    ori_data, min_val, max_val = MinMaxScaler(ori_data)

    # Network Parameters
    hidden_dim = parameters['hidden_dim']
    num_layers = parameters['num_layer']
    iterations = parameters['iterations']
    batch_size = parameters['batch_size']
    module_name = parameters['module']
    z_dim = dim
    gamma = 1

    # Embedder Network
    class Embedder(nn.Module):
        def __init__(self, hidden_dim, num_layers, dim):
            super(Embedder, self).__init__()
            self.rnn = nn.RNN(dim, hidden_dim, num_layers=num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, hidden_dim)

        def forward(self, x, t):
            #packed_x = nn.utils.rnn.pack_padded_sequence(x, t, batch_first=True, enforce_sorted=False)
            output, h_n = self.rnn(x)
            H = self.fc(output)  # Extract the output of the last layer
            return H
    # Recovery Network
    class Recovery(nn.Module):
        def __init__(self, hidden_dim, num_layers, dim):
            super(Recovery, self).__init__()
            self.rnn = nn.RNN(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, dim)

        def forward(self, h, t):
            #packed_h = nn.utils.rnn.pack_padded_sequence(h, t, batch_first=True, enforce_sorted=False)
            r_outputs, _ = self.rnn(h)
            X_tilde = self.fc(r_outputs)  # Extract the output of the last layer
            return X_tilde

    # Generator Network
    class Generator(nn.Module):
        def __init__(self, hidden_dim, num_layers, dim):
            super(Generator, self).__init__()
            self.rnn = nn.RNN(dim, hidden_dim, num_layers=num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, hidden_dim)

        def forward(self, z, t):
            z = z.to(self.fc.weight.dtype) 
            #packed_z = nn.utils.rnn.pack_padded_sequence(z, t, batch_first=True, enforce_sorted=False)
            output, g_last_states = self.rnn(z)
            E = self.fc(output)  # Extract the output of the last layer
            return E

    # Supervisor Network
    class Supervisor(nn.Module):
        def __init__(self, hidden_dim, num_layers, dim):
            super(Supervisor, self).__init__()
            self.rnn = nn.RNN(hidden_dim, hidden_dim, num_layers=num_layers-1, batch_first=True)
            self.fc = nn.Linear(hidden_dim, hidden_dim)

        def forward(self, h, t):
            #packed_h = nn.utils.rnn.pack_padded_sequence(h, t, batch_first=True, enforce_sorted=False)
            output, s_last_states = self.rnn(h)
            S = self.fc(output)  # Extract the output of the last layer
            return S

    # Discriminator Network
    class Discriminator(nn.Module):
        def __init__(self, hidden_dim, num_layers):
            super(Discriminator, self).__init__()
            self.rnn = nn.RNN(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, 1)

        def forward(self, h, t):
            #packed_h = nn.utils.rnn.pack_padded_sequence(h, t, batch_first=True, enforce_sorted=False)
            output, d_last_states = self.rnn(h)
            Y_hat = self.fc(output)  # Extract the output of the last layer
            return Y_hat

    # Convert Numpy arrays to PyTorch tensors
    ori_data = torch.FloatTensor(ori_data)
    ori_time = torch.LongTensor(ori_time)

    # Create models
    embedder = Embedder(hidden_dim, num_layers, dim)
    recovery = Recovery(hidden_dim, num_layers, dim)
    generator = Generator(hidden_dim, num_layers, dim)
    supervisor = Supervisor(hidden_dim, num_layers, dim)
    discriminator = Discriminator(hidden_dim, num_layers)

    # Loss functions
    criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss
    mse_loss = nn.MSELoss()  # Mean squared error loss

    # Optimizers
    embedder_optimizer = optim.Adam(list(embedder.parameters()) + list(recovery.parameters()))
    generator_optimizer = optim.Adam(list(generator.parameters()) + list(supervisor.parameters()))
    discriminator_optimizer = optim.Adam(discriminator.parameters())

    # Training
    print('Start Embedding Network Training')
    for itt in range(iterations):
        embedder_optimizer.zero_grad()

        # Set mini-batch
        X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
        X_mb = torch.stack([torch.tensor(x) for x in X_mb])
        T_mb = torch.stack([torch.tensor(t) for t in T_mb])

        # Embedder training
        H = embedder(X_mb, T_mb)

        H = torch.tensor(H)

        X_tilde = recovery(H, T_mb)
        E_loss_T0 = mse_loss(X_mb, X_tilde)
        E_loss0 = torch.sqrt(E_loss_T0)
        E_loss = E_loss0 + 0.1 * torch.sqrt(mse_loss(H[:, 1:, :], supervisor(H[:, :-1, :], T_mb)))

        E_loss.backward()
        embedder_optimizer.step()

        # Checkpoint
        if itt % 10 == 0:
            print('step: ' + str(itt) + '/' + str(iterations) + ', e_loss: ' + str(np.round(E_loss0.item(), 4)))

    print('Finish Embedding Network Training')

    print('Start Training with Supervised Loss Only')
    for itt in range(iterations):
        generator_optimizer.zero_grad()

        # Set mini-batch
        X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)

        # Random vector generation
        Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)

        X_mb = torch.stack([torch.tensor(x) for x in X_mb])
        T_mb = torch.stack([torch.tensor(t) for t in T_mb])
        Z_mb = torch.stack([torch.tensor(z) for z in Z_mb])

        # Generator training
        E_hat = generator(Z_mb, T_mb)
        H_hat = supervisor(E_hat, T_mb)

        G_loss_S = torch.sqrt(mse_loss(H[:, 1:, :], H_hat[:, :-1, :]))
        G_loss_S.backward()

        generator_optimizer.step()

        # Checkpoint
        if itt % 1000 == 0:
            print('step: ' + str(itt) + '/' + str(iterations) + ', s_loss: ' + str(np.round(G_loss_S.item(), 4)))

    print('Finish Training with Supervised Loss Only')

    print('Start Joint Training')
    for itt in range(iterations):
        # Generator training (twice more than discriminator training)
        for kk in range(2):
            generator_optimizer.zero_grad()

            # Set mini-batch
            X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)

            # Random vector generation
            Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)

            X_mb = torch.stack([torch.tensor(x) for x in X_mb])
            T_mb = torch.stack([torch.tensor(t) for t in T_mb])
            Z_mb = torch.stack([torch.tensor(z) for z in Z_mb])

            # Generator training
            E_hat = generator(Z_mb, T_mb)
            H_hat = supervisor(E_hat, T_mb)
            H_hat_supervise = supervisor(H, T_mb)
            X_hat = recovery(H_hat, T_mb)

            G_loss_U = criterion(discriminator(H_hat, T_mb), torch.ones((batch_size,max_seq_len, 1)))
            G_loss_U_e = criterion(discriminator(E_hat, T_mb), torch.ones((batch_size,max_seq_len, 1)))
            G_loss_S = torch.sqrt(mse_loss(H[:, 1:, :], H_hat_supervise[:, :-1, :]))
            G_loss_V1 = torch.mean(torch.abs(torch.sqrt(torch.var(X_hat, dim=0) + 1e-6)
                                              - torch.sqrt(torch.var(ori_data, dim=0) + 1e-6)))
            G_loss_V2 = torch.mean(torch.abs(torch.mean(X_hat, dim=0) - torch.mean(ori_data, dim=0)))
            G_loss_V = G_loss_V1 + G_loss_V2
            G_loss = G_loss_U + gamma * G_loss_U_e + 100 * G_loss_S + 100 * G_loss_V

            G_loss.backward()
            generator_optimizer.step()

            embedder_optimizer.zero_grad()

            # Train embedder
            E_hat = generator(Z_mb, T_mb)
            H_hat = supervisor(E_hat, T_mb)
            H_hat_supervise = supervisor(H, T_mb)
            X_hat = recovery(H_hat, T_mb)

            E_loss_T0 = mse_loss(X_mb, X_hat)
            E_loss = 10 * torch.sqrt(E_loss_T0) + 0.1 * torch.sqrt(mse_loss(H[:, 1:, :], H_hat_supervise[:, :-1, :]))

            E_loss.backward()
            embedder_optimizer.step()

        discriminator_optimizer.zero_grad()

        # Set mini-batch
        X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)

        # Random vector generation
        Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)

        X_mb = torch.stack([torch.tensor(x) for x in X_mb])
        T_mb = torch.stack([torch.tensor(t) for t in T_mb])
        Z_mb = torch.stack([torch.tensor(z) for z in Z_mb])

        # Discriminator training
        D_loss_real = criterion(discriminator(H, T_mb), torch.ones((batch_size,max_seq_len, 1)))
        D_loss_fake = criterion(discriminator(H_hat.detach(), T_mb), torch.zeros((batch_size,max_seq_len, 1)))
        D_loss_fake_e = criterion(discriminator(E_hat.detach(), T_mb), torch.zeros((batch_size,max_seq_len, 1)))
        D_loss = D_loss_real + D_loss_fake + gamma * D_loss_fake_e

        D_loss.backward()
        discriminator_optimizer.step()

        # Print multiple checkpoints
        if itt % 1000 == 0:
            print('step: ' + str(itt) + '/' + str(iterations) +
                  ', d_loss: ' + str(np.round(D_loss.item(), 4)) +
                  ', g_loss_u: ' + str(np.round(G_loss_U.item(), 4)) +
                  ', g_loss_s: ' + str(np.round(G_loss_S.item(), 4)) +
                  ', g_loss_v: ' + str(np.round(G_loss_V.item(), 4)) +
                  ', e_loss_t0: ' + str(np.round(torch.sqrt(E_loss_T0).item(), 4)))

    print('Finish Joint Training')

    # Synthetic data generation
    Z_mb = random_generator(no, z_dim, ori_time, max_seq_len)
    
    Z_mb = torch.stack([torch.tensor(z) for z in Z_mb])

    generated_data_curr = recovery(generator(Z_mb, ori_time), ori_time).detach().numpy()

    generated_data = []

    for i in range(no):
        temp = generated_data_curr[i, :ori_time[i], :]
        generated_data.append(temp)

    # Renormalization
    generated_data = generated_data * max_val
    generated_data = generated_data + min_val

    return generated_data
