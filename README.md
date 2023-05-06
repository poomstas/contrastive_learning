# Contrastive Learning on Point Cloud Dataset ShapeNet
Applying SimCLRv2 on point cloud dataset to learn how contrastive learning works

- The goal of this project is to implement a self-supervised representational learning of 3D shapes. 
- This is achieved through the use of SimCLRv2, which was originally used on 2D image datasets. 
- The output of the dataset can be used for downstream tasks such as clustering, fine-tuning, and outlier detection.
- I use the ShapeNet dataset, which comes with class labels, but for this project, I will pretend the labels don't exist.

## Set up environment
`./txt/installation_instructions.txt` file contains the conda install commands required to run the project.

## Download ShapeNet Dataset
Once the environment is set up, run the `./data/download_shapenet_data.py` file to download the ShapeNet dataset to `./data/ShapeNet`. New directories will be created.
