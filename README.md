# Contrastive Learning on Point Cloud Dataset ShapeNet
Standard supervised learning using deep learning models requires a large amount of labeled datasets, the construction of which can be costly in time and labor. Many unsupervised learning methods are introduced to help overcome the challenge. 

`SimCLR`, described in the Google AI paper "A Simple Framework for Contrastive Learning of Visual Representations" is one of many unsupervised learning approaches that can help reduce such cost.

`SimCLR` was originally used with image datasets for visual representations; in this work, I make use of `SimCLR` and `Dynamic Graph CNN` (*Dynamic Graph CNN for Learning on Point Clouds*) to acquire an unsupervised representation of objects in a point cloud dataset (`ShapeNet`), and compare the results to actual class labels.


# Project Overview
## Objectives
- Implement a self-supervised representational learning of 3D shapes represented as point cloud data.
- Compare the results of self-supervised learning with the object labels to see if they coincide well.

## Implementational Details
- I use [`SimCLRv2`](https://arxiv.org/abs/2002.05709), which was originally used on 2D image datasets, and 
- [`ShapeNet`](https://shapenet.org/) dataset, which contains 3D point cloud of various objects with class labels. 
- To extract latent features from point clouds, I use the [`EdgeConv` Model](https://arxiv.org/abs/1801.07829). 
- Using PyTorch Lightning for fast implementation.
- During training, I pretend that the class labels don't exist.


The trained model's output can be used for downstream tasks such as clustering, fine-tuning, and outlier detection.

## Feature Extraction from Point Clouds
`EdgeConv` model proposed by the paper *Dynamic Graph CNN for Learning on Point Clouds* is used to extract key features from point cloud data. A simple implementation is available in `torch_geometric`.

`EdgeConv` model is used as a replacement to `ResNet-50` base encoder in the original `SimCLR` paper, as the data type used in this work is point cloud, not image.

`EdgeConv` model is selected because of its unique advantages, some of which include:
- flexible feature learning
- robust to point cloud perturbations
- capable of end-to-end learning

Alternatives to the `EdgeConv` model include include `PointNet`, `PointNet++`, `PointTransformer`, and `SE(3)-Transformers`.


## Contrastive Learning: the SimCLR Model
The model architecture used for `SimCLR` is visualized below (from the original pape).

<p align='center'>
    <img src='/README_imgs/SimCLR_model.png' width='600' title='SimCLR Model Architecture'>
</p>


Model is implemented in `./src/model.py` file. 

```
import pytorch
```

## Contrastive Learning: the Loss Function


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

## Results
Seeing the results is straightforward. The trained model, in its inference mode, will convert a point cloud dataset into 

**Figure: t-SNE 2D Representation of Extracted Features**
<p align='center'>
    <img src='/README_imgs/tsne_results.png' width='600' title='t-SNE Representation of Data Projected to Feature Space'>
</p>

# References
- [YouTube: Deep Learning on Point Clouds](https://youtu.be/gm_oW0bdzHs)
- [SimCLR Explained](https://youtu.be/APki8LmdJwY)
- [Google AI Post on SimCLR](https://ai.googleblog.com/2020/04/advancing-self-supervised-and-semi.html)
- [SimCLR Paper](https://arxiv.org/abs/2002.05709)

TODO: Do supervised point cloud classification and compare results. See: [Video](https://www.youtube.com/watch?v=ctdi4Fjp_50&t=21s&ab_channel=ConnorShorten)