# %%
import torch
import torch.nn as nn
from torch.nn import Linear
from torch_geometric.nn import MLP, DynamicEdgeConv, global_max_pool

# %%
class SimCLRPointCloud(nn.Module):
    def __init__(self, augmentation, k=20, aggregation='max'):
        super().__init__()
        self.augmentation = augmentation                                    # Define augmentations. Has to be probabilistic.
        self.conv1 = DynamicEdgeConv(MLP([2*3, 64, 64]), k, aggregation)    # Feature Extraction 1
        self.conv2 = DynamicEdgeConv(MLP([2*64, 128]), k, aggregation)      # Feature Extraction 2
        self.lin = Linear(128 + 64, 128)                                    # Encoder Head
        self.mlp = MLP([128, 256, 32], norm=None)                           # Projection head

    def forward(self, data, train=True):
        if train:
            # Get 2 augmentations of the batch
            augm_1 = self.augmentation(data)
            augm_2 = self.augmentation(data)

            # Extract properties
            pos1, batch1 = augm_1.pos, augm_1.batch
            pos2, batch2 = augm_2.pos, augm_2.batch

            # Get representations for first augmented view
            x1 = self.conv1(pos1, batch1)
            x2 = self.conv2(x1, batch1)
            h_points_1 = self.lin(torch.cat([x1, x2], dim=1))

            # Get representations for second augmented view
            x1 = self.conv1(pos2, batch2)
            x2 = self.conv2(x1, batch2)
            h_points_2 = self.lin(torch.cat([x1, x2], dim=1))
            
            # Global representation
            h1 = global_max_pool(h_points_1, batch1)
            h2 = global_max_pool(h_points_2, batch2)

            # Transformation for loss function
            compact_h1 = self.mlp(h1)
            compact_h2 = self.mlp(h2)
            return h1, h2, compact_h1, compact_h2

        else:
            x1 = self.conv1(data.pos, data.batch)
            x2 = self.conv2(x1, data.batch)
            h_points = self.lin(torch.cat([x1, x2], dim=1))
            return global_max_pool(h_points, data.batch)


# %%
if __name__=='__main__':
    import torch_geometric.transforms as T
    from torch_geometric.datasets import ShapeNet

    dataset = ShapeNet(root='../data/ShapeNet/', categories=['Table', 'Lamp', 'Guitar', 'Motorbike']).shuffle()[:5000]

    augmentation = T.Compose([T.RandomJitter(0.005), T.RandomFlip(1), T.RandomShear(0.3)])
    model = SimCLRPointCloud(augmentation=augmentation)

    h1, h2, compact_h1, compact_h2 = model(dataset[0])
    
    print('h1: ', h1)
    print('h2: ', h2)
    print('compact_h1: ', compact_h1)
    print('compact_h2: ', compact_h2)
