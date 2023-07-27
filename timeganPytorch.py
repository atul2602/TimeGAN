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
            sorted_lengths, sorted_idx = torch.sort(t, descending=True)
            sorted_x = x[sorted_idx]
            packed_x = nn.utils.rnn.pack_padded_sequence(sorted_x, sorted_lengths, batch_first=True, enforce_sorted=False)
            output, _ = self.rnn(packed_x)
            padded_output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
            _, unsorted_idx = torch.sort(sorted_idx)
            H = padded_output[unsorted_idx]
            H = self.fc(H)
            return H


    class Recovery(nn.Module):
        def __init__(self, hidden_dim, num_layers, dim):
            super(Recovery, self).__init__()
            self.rnn = nn.RNN(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, dim)

        def forward(self, h, t):
            sorted_lengths, sorted_idx = torch.sort(t, descending=True)
            sorted_h = h[sorted_idx]
            packed_h = nn.utils.rnn.pack_padded_sequence(sorted_h, sorted_lengths, batch_first=True, enforce_sorted=False)
            output, _ = self.rnn(packed_h)
            padded_output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
            _, unsorted_idx = torch.sort(sorted_idx)
            X_tilde = self.fc(padded_output[unsorted_idx])
            return X_tilde


    class Generator(nn.Module):
        def __init__(self, hidden_dim, num_layers, dim):
            super(Generator, self).__init__()
            self.rnn = nn.RNN(dim, hidden_dim, num_layers=num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, hidden_dim)

        def forward(self, z, t):
            print("Data types:")
            print("z dtype:", z.dtype)
            print("t dtype:", t.dtype)
            sorted_lengths, sorted_idx = torch.sort(t, descending=True)
            sorted_z = z[sorted_idx]
            packed_z = nn.utils.rnn.pack_padded_sequence(sorted_z, sorted_lengths, batch_first=True, enforce_sorted=False)
            output, _ = self.rnn(packed_z)
            padded_output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
            _, unsorted_idx = torch.sort(sorted_idx)
            G = self.fc(padded_output[unsorted_idx])
            print("G shape is: ", G.shape)
            return G
        
    class Supervisor(nn.Module):
        def __init__(self, hidden_dim, num_layers, dim):
            super(Supervisor, self).__init__()
            self.rnn = nn.RNN(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, hidden_dim)

        def forward(self, h, t):
            print("Hidden dimension is: ", h.shape)
            sorted_lengths, sorted_idx = torch.sort(t, descending=True)
            sorted_h = h[sorted_idx]
            packed_h = nn.utils.rnn.pack_padded_sequence(sorted_h, sorted_lengths, batch_first=True, enforce_sorted=False)

            output, _ = self.rnn(packed_h)
            padded_output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
            _, unsorted_idx = torch.sort(sorted_idx)
            S = padded_output[unsorted_idx]
            S = self.fc(S)
            return S
    
    class Discriminator(nn.Module):
        def __init__(self, hidden_dim, num_layers):
            super(Discriminator, self).__init__()
            self.rnn = nn.RNN(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, 1)

        def forward(self, h, t):
            sorted_lengths, sorted_idx = torch.sort(t, descending=True)
            sorted_h = h[sorted_idx]
            packed_h = nn.utils.rnn.pack_padded_sequence(sorted_h, sorted_lengths, batch_first=True, enforce_sorted=False)
            output, _ = self.rnn(packed_h)
            padded_output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
            _, unsorted_idx = torch.sort(sorted_idx)
            Y_hat = self.fc(padded_output[unsorted_idx])
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

        # Set mini-batch
        X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
        X_mb = torch.stack([torch.tensor(x) for x in X_mb])  # Convert X_mb to PyTorch tensor
        T_mb = torch.stack([torch.tensor(t) for t in T_mb]) 

        # Zero the gradients
        embedder_optimizer.zero_grad()

        # Embedder forward pass
        H = embedder(X_mb, T_mb)

        # Calculate the reconstruction loss
        X_tilde = recovery(H, T_mb)
        E_loss_T0 = nn.MSELoss()(X_mb, X_tilde)
        E_loss0 = torch.sqrt(E_loss_T0)
        E_loss = E_loss0 + 0.1 * torch.sqrt(nn.MSELoss()(H[:, 1:, :], H[:, :-1, :]))

        # Backpropagation and optimization
        E_loss.backward()
        embedder_optimizer.step()

        # Checkpoint
        if itt % 1000 == 0:
            print('step: ' + str(itt) + '/' + str(iterations) + ', e_loss: ' + str(torch.sqrt(E_loss_T0).item()))
    print('Finish Embedding Network Training')

    with torch.no_grad():
        H_supervise = embedder(X_mb, T_mb)  # Generate H for supervised loss later

    print('Start Training with Supervised Loss Only')
    for itt in range(iterations):
        generator_optimizer.zero_grad()

        # Set mini-batch
        X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)

        # Random vector generation
        Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)

        X_mb = torch.stack([torch.tensor(x, dtype = torch.float32) for x in X_mb])
        T_mb = torch.stack([torch.tensor(t, dtype = torch.int64) for t in T_mb])
        Z_mb = torch.stack([torch.tensor(z, dtype = torch.float32) for z in Z_mb])
        print("X_mb shape is: ", X_mb.shape)
        print("T_mb shape is: ", T_mb.shape)
        print("Z_mb shape is: ", Z_mb.shape)


        E_hat = generator(Z_mb, T_mb)
        H_hat = supervisor(E_hat, T_mb)
        G_loss_S = torch.sqrt(mse_loss(H_hat[:, 1:, :], H_hat[:, :-1, :]))

        # Calculate the supervised loss using the stored H
        G_loss_S_supervised = torch.sqrt(mse_loss(H_hat[:, 1:, :], H_supervise[:, :-1, :]))

        # Calculate the total G_loss_S using both supervised and unsupervised components
        G_loss_S_total = G_loss_S + G_loss_S_supervised

        # Backpropagate with the combined loss
        G_loss_S_total.backward()
        generator_optimizer.step()

        # Checkpoint
        if itt % 1000 == 0:
            print('step: ' + str(itt) + '/' + str(iterations) + ', s_loss: ' + str(np.round(G_loss_S.item(), 4)))

    print('Finish Training with Supervised Loss Only')

    with torch.no_grad():
        H_supervise = embedder(X_mb, T_mb)  # Generate H for supervised loss later

    print('Start Joint Training')
    for itt in range(iterations):
        # Generator training (twice more than discriminator training)
        for kk in range(2):
            # confused about the H_supervise in this part.
            generator_optimizer.zero_grad()

            # Set mini-batch
            X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)

            # Random vector generation
            Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)

            X_mb = torch.stack([torch.tensor(x, dtype = torch.float32) for x in X_mb])
            T_mb = torch.stack([torch.tensor(t, dtype = torch.int64) for t in T_mb])
            Z_mb = torch.stack([torch.tensor(z, dtype = torch.float32) for z in Z_mb])

            # Generator training
            E_hat = generator(Z_mb, T_mb)
            H_hat = supervisor(E_hat, T_mb)
            H_hat_supervise = supervisor(H_supervise, T_mb) 
            X_hat = recovery(H_hat, T_mb)

            G_loss_U = criterion(discriminator(H_hat, T_mb), torch.ones((batch_size,max_seq_len, 1)))
            G_loss_U_e = criterion(discriminator(E_hat, T_mb), torch.ones((batch_size,max_seq_len, 1)))
            G_loss_S = torch.sqrt(mse_loss(H_supervise[:, 1:, :], H_hat_supervise[:, :-1, :]))
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
            H_hat_supervise = supervisor(H_supervise, T_mb)
            X_hat = recovery(H_hat, T_mb)

            E_loss_T0 = mse_loss(X_mb, X_hat)
            E_loss = 10 * torch.sqrt(E_loss_T0) + 0.1 * torch.sqrt(mse_loss(H_supervise[:, 1:, :], H_hat_supervise[:, :-1, :]))

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
        D_loss_real = criterion(discriminator(H_supervise, T_mb), torch.ones((batch_size,max_seq_len, 1)))
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
    
    Z_mb = torch.stack([torch.tensor(z, dtype=torch.float32) for z in Z_mb])
    ori_time = torch.stack([torch.tensor(t, dtype=torch.int64) for t in ori_time])

    generated_data_curr = recovery(generator(Z_mb, ori_time), ori_time).detach().numpy()

    generated_data = []

    for i in range(no):
        temp = generated_data_curr[i, :ori_time[i], :]
        generated_data.append(temp)

    # Renormalization
    generated_data = generated_data * max_val
    generated_data = generated_data + min_val

    return generated_data
