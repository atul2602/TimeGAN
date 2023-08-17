import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch

def visualization(ori_data, generated_data, analysis):
    anal_sample_no = min([1000, len(ori_data)])
    idx = np.random.permutation(len(ori_data))[:anal_sample_no]
    
    ori_data = np.asarray(ori_data)
    generated_data = np.asarray(generated_data)  
    
    ori_data = ori_data[idx]
    generated_data = generated_data[idx]
    
    no, seq_len, dim = ori_data.shape  
    
    prep_data = torch.tensor(np.mean(ori_data, axis=2), dtype=torch.float32)
    prep_data_hat = torch.tensor(np.mean(generated_data, axis=2), dtype=torch.float32)
    
    colors = ["red" for _ in range(anal_sample_no)] + ["blue" for _ in range(anal_sample_no)]    
    
    if analysis == 'pca':
        pca = PCA(n_components=2)
        pca_results = pca.fit_transform(prep_data)
        pca_hat_results = pca.transform(prep_data_hat)
        
        f, ax = plt.subplots(1)    
        plt.scatter(pca_results[:, 0], pca_results[:, 1],
                    c=colors[:anal_sample_no], alpha=0.2, label="Original")
        plt.scatter(pca_hat_results[:, 0], pca_hat_results[:, 1], 
                    c=colors[anal_sample_no:], alpha=0.2, label="Synthetic")
      
        ax.legend()  
        plt.title('PCA plot')
        plt.xlabel('x-pca')
        plt.ylabel('y_pca')
        plt.show()
        
    elif analysis == 'tsne':
        prep_data_final = torch.cat((prep_data, prep_data_hat), dim=0)
        
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(prep_data_final.numpy())
          
        f, ax = plt.subplots(1)
          
        plt.scatter(tsne_results[:anal_sample_no, 0], tsne_results[:anal_sample_no, 1], 
                    c=colors[:anal_sample_no], alpha=0.2, label="Original")
        plt.scatter(tsne_results[anal_sample_no:, 0], tsne_results[anal_sample_no:, 1], 
                    c=colors[anal_sample_no:], alpha=0.2, label="Synthetic")
      
        ax.legend()
          
        plt.title('t-SNE plot')
        plt.xlabel('x-tsne')
        plt.ylabel('y_tsne')
        plt.show()
