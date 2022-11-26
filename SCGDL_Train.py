import numpy as np
import pandas as pd
from tqdm import tqdm 
import scipy.sparse as sp

from SCGDL import ResGatedGraphmodel
from SCGDL_auxiliary import Adata2Torch_data

import torch
import torch.backends.cudnn as cudnn
cudnn.deterministic = True 
cudnn.benchmark = True 
import torch.nn.functional as F 
from torch_geometric.nn import DeepGraphInfomax

class my_data():
    def __init__(self, x, edge_index, edge_attr):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr

def corruption(data): 
    x = data.x[torch.randperm(data.x.size(0))] 
    return my_data(x, data.edge_index, data.edge_attr)


def SCGDL_Train(adata, 
                hidden_dims=[128, 128], 
                num_epochs=1000, 
                lr=1e-6, 
                key_added='SCGDL',
                gradient_clipping=5., 
                weight_decay=0.0001, 
                random_seed=0, save_loss=True):
    """\
    Training graph attention auto-encoder.
    Parameters
    ----------
    adata: AnnData object of scanpy package.
    hidden_dims: The dimension of the encoder.
    n_epochs:Number of total epochs for training.
    lr: Learning rate for AdamOptimizer.
    key_added: The latent embeddings are saved in adata.obsm[key_added].
    gradient_clipping: Gradient Clipping.
    weight_decay: Weight decay for AdamOptimizer.
    save_loss: If True, the training loss is saved in adata.uns['SCGDL_loss'].
    save_reconst_exp: If True, the reconstructed expression profiles are saved in adata.layers['SCGDL_ReX'].
    device: See torch.device.

    Returns
    -------
    AnnData
    """
    seed=random_seed
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    adata.X = sp.csr_matrix(adata.X)
    if "Spatial_highly_variable_genes" in adata.var.columns:
        adata_Vars =  adata[:, adata.var['Spatial_highly_variable_genes']]
        print('Input Size using Spatial_variable_genes: ', adata_Vars.shape)
    elif 'highly_variable' in adata.var.columns:
        adata_Vars =  adata[:, adata.var['highly_variable']]
        print('Input Size using Highly_variable_genes: ', adata_Vars.shape)
    else:
        adata_Vars = adata
        print('Input Size using All genes list: ', adata_Vars.shape) 

    if 'Spatial_Net' not in adata.uns.keys():
        raise ValueError("Please Compute Spatial Network using Spatial_Dis_Cal function first!") 

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data = Adata2Torch_data(adata_Vars) 
    hidden_dims = [data.x.shape[1]] + hidden_dims 
  
    DGI_model = DeepGraphInfomax(
        hidden_channels=hidden_dims[1], 
        encoder=ResGatedGraphmodel(hidden_dims), 
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)), 
        corruption=corruption).to(device) 
    DGI_optimizer = torch.optim.Adam(DGI_model.parameters(), lr=lr, weight_decay=weight_decay)
    data = data.to(device)

    import time
    start_time = time.time()
    loss_list = []
    for epoch in tqdm(range(1, num_epochs+1)):
        DGI_model.train()
        DGI_optimizer.zero_grad() 
        pos_z, neg_z, summary = DGI_model(data=data) 
        DGI_loss = DGI_model.loss(pos_z, neg_z, summary) 
        loss_list.append(DGI_loss.item())
        DGI_loss.backward()
        torch.nn.utils.clip_grad_norm_(DGI_model.parameters(), gradient_clipping) 
        DGI_optimizer.step()
        if ((epoch)%1000) == 0:
            print('Epoch: {:03d}, Loss: {:.4f}'.format(epoch, np.mean(loss_list)))
    end_time = time.time()
    print('Elapsed training time:{:.4f} seconds'.format((end_time-start_time)))

    DGI_model.eval()
    pos_z, neg_z, summary = DGI_model(data=data) 

    SCGDL_rep = pos_z.to('cpu').detach().numpy() 
    adata.obsm[key_added] = SCGDL_rep

    if save_loss:
        adata.uns['SCGDL_loss'] = loss_list

    return adata
