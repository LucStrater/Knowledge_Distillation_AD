# Coding Assignment: Reproducing Salehi et al. with Transformers instead of VGG

This repository is forked from the official implementation of [Multiresolution Knowledge Distillation for Anomaly Detection](https://arxiv.org/pdf/2011.11108.pdf). The code-base is adjusted to work on google colab and with the corresponding packages. The base model (VGG) and the transformer based adaption are tested only on CIFAR10 dataset. To train and test the base model, use the notebook called vgg_kdad.ipynb . The base model takes extremely long to train so I provided checkpoints in the outputs/ folder. To train and test the vision transformer based adpation, use the notebook called vit_kdad.ipynb .




