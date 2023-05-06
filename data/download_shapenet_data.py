from torch_geometric.datasets import ShapeNet

dataset = ShapeNet(root='./ShapeNet', categories=['Table', 'Lamp', 'Guitar', 'Motorbike']).shuffle()[:5000]