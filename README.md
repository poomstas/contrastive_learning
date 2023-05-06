# Contrastive Learning on Point Cloud Dataset ShapeNet
Applying SimCLRv2 on point cloud dataset to learn how contrastive learning works

- The goal of this project is to implement a self-supervised representational learning of 3D shapes. 
- This is achieved through the use of SimCLRv2, which was originally used on 2D image datasets. 
- The output of the dataset can be used for downstream tasks such as clustering, fine-tuning, and outlier detection.
- I use the ShapeNet dataset, which comes with class labels, but for this project, I will pretend the labels don't exist.

## Environment Setup
`./txt/installation_instructions.txt` file contains the conda install commands required to run the project.

## Download ShapeNet Dataset
Once the environment is set up, run the `./data/download_shapenet_data.py` file to download the ShapeNet dataset to `./data/ShapeNet`. New directories will be created.

## ShapeNet EDA
Some sample visualizations of the original data and augmented data is available in `/experiments/EDA.py`. See next session for information on augmentations.


**Figure: Sample Point Cloud Dataset of a Table**
<p align='center'>
    <img src='/README_imgs/Table.png' width='450' title='Point Cloud of Sample Table'>
</p>

**Figure: Data Augmentation (Jitter, Flip, Shear) Applied to Table Point Cloud**
<p align='center'>
    <img src='/README_imgs/TableAugmented.png' width='450' title='Point Cloud of Sample Table with Augmentations'>
</p>

**Figure: Sample Point Cloud Dataset of a Motorbike**
<p align='center'>
    <img src='/README_imgs/Motorbike.png' width='450' title='Point Cloud of Sample Motorbike'>
</p>

**Figure: Data Augmentation (Jittering, Shifting, Shearing) Applied to Table Point Cloud**
<p align='center'>
    <img src='/README_imgs/MotorbikeAugmented.png' width='450' title='Point Cloud of Sample Motorbike with Augmentations'>
</p>

