# %%
'''
    Download the ShapeNet dataset, take a subset, and take a look at it to see how the data is structured.
'''

# %%
import random
import plotly.express as px
from torch_geometric.datasets import ShapeNet

# %% Load the ShapeNet dataset and take subset
''' 
    pos: Normalized positions as 3D coordinates 
    x : Normal vectors (unused in this study)
    y : ~~ (unused in this study)
    category: ~~ (used later to validate)
'''

dataset = ShapeNet(root='../data/ShapeNet/', categories=['Table', 'Lamp', 'Guitar', 'Motorbike']).shuffle()[:5000]
print(dataset)
print('Sample: ', dataset[0])

# %% Plot 3D point cloud
def plot_3d_shape(shape):
    print('Number of data points: ', shape.x.shape[0])
    x = shape.pos[:, 0]
    y = shape.pos[:, 1]
    z = shape.pos[:, 2]
    fig = px.scatter_3d(x=x, y=y, z=z, opacity=0.3)
    fig.show()

sample_idx = random.choice(range(len(dataset)))
plot_3d_shape(dataset[sample_idx])

# %% Check the distribution of each class
cat_dict = {key: 0 for key in dataset.categories}
for datapoint in dataset:
    cat_dict[dataset.categories[datapoint.category.int()]] += 1
print(cat_dict)
