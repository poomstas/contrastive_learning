''' Make sure to set up the environment before running this script. See ../txt/requirements.txt '''
from torch_geometric.datasets import ShapeNet

dataset = ShapeNet(root='./ShapeNet', categories=['Table', 'Lamp', 'Guitar', 'Motorbike']).shuffle()[:5000]