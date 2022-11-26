import pandas as pd
import numpy as np
import sklearn.neighbors
from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics.cluster import adjusted_rand_score
import scipy.sparse as sp
import seaborn as sns
import matplotlib.pyplot as plt

import torch
from torch_geometric.data import Data

def Adata2Torch_data(adata): 
    G_df = adata.uns['Spatial_Net'].copy() 
    spots = np.array(adata.obs_names) 
    spots_id_tran = dict(zip(spots, range(spots.shape[0]))) 
    G_df['Spot1'] = G_df['Spot1'].map(spots_id_tran) 
    G_df['Spot2'] = G_df['Spot2'].map(spots_id_tran) 

    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Spot1'], G_df['Spot2'])), shape=(adata.n_obs, adata.n_obs))
    G = G + sp.eye(G.shape[0])

    edgeList = np.nonzero(G) 
    if type(adata.X) == np.ndarray:
        data = Data(edge_index=torch.LongTensor(np.array(
            [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X))  
    else:
        data = Data(edge_index=torch.LongTensor(np.array(
            [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X.todense()))  
    return data

def Spatial_Dis_Cal(adata, rad_dis=None, knn_dis=None, model='Radius', verbose=True):
    """\
    Calculate the spatial neighbor networks, as the distance between two spots.
    Parameters
    ----------
    adata:  AnnData object of scanpy package.
    rad_dis:  radius distance when model='Radius' 
    knn_dis:  The number of nearest neighbors when model='KNN'
    model:
        The network construction model. When model=='Radius', the spot is connected to spots whose distance is less than rad_dis. 
        When model=='KNN', the spot is connected to its first knn_dis nearest neighbors.
    Returns
    -------
    The spatial networks are saved in adata.uns['Spatial_Net']
    """
    assert(model in ['Radius', 'KNN']) 
    if verbose:
        print('------Calculating spatial graph...')
    coor = pd.DataFrame(adata.obsm['spatial']) 
    coor.index = adata.obs.index 
    coor.columns = ['Spatial_X', 'Spatial_Y'] 

    if model == 'Radius':
        nbrs = sklearn.neighbors.NearestNeighbors(radius=rad_dis).fit(coor)
        distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
        KNN_list = []
        for spot in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([spot]*indices[spot].shape[0], indices[spot], distances[spot]))) 
    
    if model == 'KNN':
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=knn_dis+1).fit(coor)
        distances, indices = nbrs.kneighbors(coor)
        KNN_list = []
        for spot in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([spot]*indices.shape[1],indices[spot,:], distances[spot,:])))

    KNN_df = pd.concat(KNN_list) 
    KNN_df.columns = ['Spot1', 'Spot2', 'Distance']

    Spatial_Net = KNN_df.copy()
    Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance']>0,]
    id_spot_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), )) 
    Spatial_Net['Spot1'] = Spatial_Net['Spot1'].map(id_spot_trans) 
    Spatial_Net['Spot2'] = Spatial_Net['Spot2'].map(id_spot_trans) 
    if verbose:
        print('The graph contains %d edges, %d spots.' %(Spatial_Net.shape[0], adata.n_obs)) 
        print('%.4f neighbors per spot on average.' %(Spatial_Net.shape[0]/adata.n_obs)) 

    adata.uns['Spatial_Net'] = Spatial_Net

def Spatial_Dis_Draw(adata):
    import matplotlib.pyplot as plt
    Num_edge = adata.uns['Spatial_Net']['Spot1'].shape[0] 
    Mean_edge = Num_edge/adata.shape[0] 
    plot_df = pd.value_counts(pd.value_counts(adata.uns['Spatial_Net']['Spot1'])) 
    plot_df = plot_df/adata.shape[0]  
    fig, ax = plt.subplots(figsize=[4,4],dpi=300)
    plt.ylabel('Percentage')
    plt.xlabel('Edge Numbers per Spot')
    plt.title('Number of Neighbors for Spots (Average=%.2f)'%Mean_edge)
    ax.bar(plot_df.index, plot_df,color="#aa40fc",edgecolor="#f7b6d2",linewidth=2)

def Cal_Spatial_variable_genes(adata):
    import SpatialDE
    counts = pd.DataFrame(adata.X, columns=adata.var_names, index=adata.obs_names)
    coor = pd.DataFrame(adata.obsm['spatial'], columns=['Spatial_X', 'Spatial_Y'], index=adata.obs_names)
    Spatial_var_genes = SpatialDE.run(coor, counts)
    Spatial_3000_var_genes = Spatial_var_genes["g"].values[0:3000]
    Spatial_3000_var_genes = pd.DataFrame(Spatial_3000_var_genes)
    all_genes = counts.columns.to_frame()
    for i in range(len(all_genes.values)):
        if all_genes.values[i] in Spatial_3000_var_genes.values:
            all_genes.values[i] =1
        else:
            all_genes.values[i] =0
    Spatial_highly_genes = all_genes.squeeze()
    adata.var["Spatial_highly_variable_genes"] = Spatial_highly_genes.astype(bool)

def DGI_loss_Draw(adata):
    import matplotlib.pyplot as plt
    if "SCGDL_loss" not in adata.uns.keys():
        raise ValueError("Please Train DGI Model using SCGDL_Train function first!") 
    Train_loss = adata.uns["SCGDL_loss"]
    plt.style.use('default') 
    plt.plot(Train_loss,label='Training loss',linewidth=2)
    plt.xlabel("Number of Epochs")
    plt.ylabel("Loss of DGI model")
    plt.legend()
    plt.grid()

def BGMM(adata,n_cluster,used_obsm='SCGDL'):
    """
    BayesianGaussianMixture for spatial clustering.
    """

    knowledge = BayesianGaussianMixture(n_components=n_cluster,
                                        weight_concentration_prior_type ='dirichlet_process', ##'dirichlet_process' or dirichlet_distribution'
                                        weight_concentration_prior = 50).fit(adata.obsm[used_obsm])                                  
    # load ground truth for ARI and NMI computation.
    Ann_df = pd.read_csv("/home/tengliu/Torch_pyG/SCGDL_Upload_Files/data/Human_DLPFC/151675_truth.txt", sep='\t', header=None, index_col=0)
    Ann_df.columns = ['Ground Truth']
    adata.obs['Ground Truth'] = Ann_df.loc[adata.obs_names, 'Ground Truth']

    method = "BayesianGaussianMixture"
    labels = knowledge.predict(adata.obsm[used_obsm])+1
    Ann_df.columns = [method] # 给dataframe类型的数据 列命名'Spectral clusters'
    Ann_df.loc[:,method] = labels #先将分类结果标签添加到导入的Ground Truth文件中
    adata.obs[method] = Ann_df.loc[adata.obs_names, method] ##将分类结果添加到obs中
    adata.obs[method] = adata.obs[method].astype('category') 
    obs_df = adata.obs.dropna() ##过滤掉缺失值的行，即当前分类值为Nan，就把该行过滤掉。
    ARI = adjusted_rand_score(obs_df[method], obs_df['Ground Truth']) ## ARI 是从 包sklearn.metrics.cluster中导入的衡量标准
    adata.uns["ARI"] = ARI 
    return adata
