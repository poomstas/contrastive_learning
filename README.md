# Contrastive Learning on Point Cloud Dataset ShapeNet
Standard supervised learning using deep learning models requires a large amount of labeled datasets, the construction of which can be costly in time and labor. Many unsupervised learning methods are introduced to help overcome the challenge. 

SimCLR, described in the Google AI paper "A Simple Framework for Contrastive Learning of Visual Representations" is one of many unsupervised learning approaches that can help reduce such cost.

SimCLR was originally used with image datasets for visual representations; in this work, I make use of SimCLR and Dynamic Graph CNN ("Dynamic Graph CNN for Learning on Point Clouds") to acquire an unsupervised representation of objects in a point cloud dataset (ShapeNet), and compare the results to actual class labels.


# Project Overview
## Objectives
- Implement a self-supervised representational learning of 3D shapes represented as point cloud data.
- Compare the results of self-supervised learning with the object labels to see if they coincide well.

## Implementational Details
- I use [SimCLRv2](https://arxiv.org/abs/2002.05709), which was originally used on 2D image datasets, and 
- [ShapeNet](https://shapenet.org/) dataset, which contains 3D point cloud of various objects with class labels. 
- To extract latent features from point clouds, I use the [EdgeConv Model](https://arxiv.org/abs/1801.07829). 
- Using PyTorch Lightning for fast implementation.
- I will pretend the labels don't exist, except for the very end where I will validate the results. 


The output of the dataset can be used for downstream tasks such as clustering, fine-tuning, and outlier detection.

## Feature Extraction from Point Clouds
EdgeConv model proposed by the paper *Dynamic Graph CNN for Learning on Point Clouds* is used to extract key features from point cloud data. A simple implementation is available in `torch_geometric`.

Alternative models include PointNet, PointNet++, PointTransformer, SE(3)-Transformers, etc.


## Contrastive Learning: SimCLR
- SimCLR (4x) (contrastive learning, without labels) was able to achieve the same classification accuracy as a ResNet-50 trained with supervised learning (with labels). [Ref]
- SimCLR (4x) did so at the cost of more parameters (375 million vs. 24 million).

## Contrastive Learning: Loss Functions

## Contrastive Learning: Model Implementation

# Running the Code
## Environment Setup
`/txt/installation_instructions.txt` file contains the conda install commands required to run the project.

## Download ShapeNet Dataset
Once the environment is set up, run the `/data/download_shapenet_data.py` file to download the ShapeNet dataset to `./data/ShapeNet`. New directories will be created.

## ShapeNet EDA
Some sample visualizations of the original data and augmented data is available in `/experiments/EDA.py`. See next session for information on augmentations.


**Figure: Sample Point Cloud Dataset of a Table, Before and After Augmentation (Jitter, Flip, Shear)**
<p align='center'>
    <img src='/README_imgs/Table.png' width='600' title='Point Cloud of Sample Table'>
</p>

**Figure: Sample Point Cloud Dataset of a Motorbike, Before and After Augmentation (Jitter, Flip, Shear)**
<p align='center'>
    <img src='/README_imgs/Motorbike.png' width='600' title='Point Cloud of Sample Motorbike'>
</p>

## Preprocess Dataset
Run the `./data/preprocess_data.py` file, which contains the script to preprocess the dataset before it can be used for SimCLRv2 contrastive learning. The preprocessing steps include:

- Store multiple data points into one Data object, using PyTorch
- Compute data augmentations ()
    - Rotation / Flip (if used layer is not rotation invariant)
    - Jittering (adding noise to the coordinates)
    - Shifting / Shearing

## Training the SimCLR Model
- Default training epochs set at 10

## Testing the SimCLR Model

# References
- [YouTube: Deep Learning on Point Clouds](https://youtu.be/gm_oW0bdzHs)
- [SimCLR Explained](https://youtu.be/APki8LmdJwY)
- [Google AI Post on SimCLR](https://ai.googleblog.com/2020/04/advancing-self-supervised-and-semi.html)
- [SimCLR Paper](https://arxiv.org/abs/2002.05709)

TODO: Do supervised point cloud classification and compare results. See: [Video](https://www.youtube.com/watch?v=ctdi4Fjp_50&t=21s&ab_channel=ConnorShorten)