# %%
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from train import TrainSimCLR
from torch_geometric.datasets import ShapeNet
from torch_geometric.loader import DataLoader
from src.paths import DATA

# %%
CKPT_PATH = '/home/brian/github/contrastive_learning_point_cloud/model_checkpoint/aorus_20230510_114317/epoch=06-loss=0.50589.ckpt'

# %%
model = TrainSimCLR.load_from_checkpoint(CKPT_PATH)

# %%
dataset = ShapeNet(root=DATA, categories=['Table', 'Lamp', 'Guitar', 'Motorbike']).shuffle()[:5000]
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# %%
# Get sample batch
sample = next(iter(dataloader)).to('cuda:0')

# Get representations
h = model(sample, train=False)
h = h.cpu().detach()
labels = sample.category.cpu().detach().numpy()
labels = [model.categories[i] for i in labels]

# Get low-dimensional t-SNE Embeddings
h_embedded = TSNE(n_components          = 2, 
                  learning_rate         = 'auto',
                  init='random').fit_transform(h.numpy())

# Plot
ax = sns.scatterplot(x                  = h_embedded[:,0], 
                     y                  = h_embedded[:,1], 
                     hue                = labels, 
                     alpha              = 0.7, 
                     palette            = 'tab10')

# Add labels to be able to identify the data points
annotations = list(range(len(h_embedded[:,0])))

def label_points(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x']+.02, point['y'], str(int(point['val'])))

label_points(pd.Series(h_embedded[:,0]), 
            pd.Series(h_embedded[:,1]), 
            pd.Series(annotations), 
            plt.gca()) 

# %%
