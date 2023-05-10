# %%
import seaborn as sns
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from torch_geometric.datasets import ShapeNet
from torch_geometric.loader import DataLoader

from train import TrainSimCLR
from src.paths import DATA
from experiments.EDA import plot_3d_shape

# %%
CKPT_PATH = '/home/brian/github/contrastive_learning_point_cloud/model_checkpoint/aorus_20230510_114317/epoch=06-loss=0.50589.ckpt'

model = TrainSimCLR.load_from_checkpoint(CKPT_PATH)
dataset = ShapeNet(root=DATA, categories=['Table', 'Lamp', 'Guitar', 'Motorbike']).shuffle()[:5000]
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# %%
sample = next(iter(dataloader)).to('cuda:0')

h = model(sample, train=False)
h = h.cpu().detach()
labels = sample.category.cpu().detach().numpy()
labels = [model.categories[i] for i in labels]

h_embedded = TSNE(n_components          = 2, 
                  learning_rate         = 'auto',
                  init='random').fit_transform(h.numpy())

ax = sns.scatterplot(x                  = h_embedded[:,0], 
                     y                  = h_embedded[:,1], 
                     hue                = labels, 
                     alpha              = 0.7, 
                     palette            = 'tab10')

annotations = list(range(len(h_embedded[:,0])))

def label_points(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x']+.02, point['y'], str(int(point['val'])))

label_points(pd.Series(h_embedded[:,0]), 
            pd.Series(h_embedded[:,1]), 
            pd.Series(annotations), 
            plt.gca()) 

# %% Calculate similarity matrix
def sim_matrix(a, b, eps=1e-8):
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n)) # eps for numerical stability
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n)) # eps for numerical stability
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

similarity  = sim_matrix(h, h)
max_indices = torch.topk(similarity, k=2)[1][:, 1]
max_vals    = torch.topk(similarity, k=2)[0][:, 1]

# Select index
indx        = 12 # 12 23 43
similar_indx= max_indices[indx]
print(f'Most similar data point in the embedding space for {indx} is {similar_indx}')

plot_3d_shape(sample[indx].cpu())
plot_3d_shape(sample[similar_indx].cpu())
