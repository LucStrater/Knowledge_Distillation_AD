# Coding Assignment: Reproducing Salehi et al. with Transformers instead of VGG

This repository is forked from the official implementation of [Multiresolution Knowledge Distillation for Anomaly Detection](https://arxiv.org/pdf/2011.11108.pdf). The code-base is adjusted to work on google colab and with the corresponding packages. The base model (VGG) and the transformer based adaption are tested only on CIFAR10 dataset. To train and test the base model, use the notebook called vgg_kdad.ipynb . The base model takes extremely long to train so I provided checkpoints in the outputs/ folder. To train and test the vision transformer based adpation, use the notebook called vit_kdad.ipynb .

### Result base model (VGG)
Training the base model took extremely long (2 hours) and due to colab I could not run things in the background, thus I did only one run/seed. The result of that run can be found in vgg_kdad.ipynb ,  but in summary I trained for 200 epochs, 1e-3 learning rate, 64 batch size. I got an RocAUC of 77.03%, which is on the low end of the runs in the original paper, but seems reasonable enough to not investigate much further. 

### Results transformer based model (VIT)
TBA




