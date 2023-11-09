# Coding Project: Reproducing Salehi et al. with ViT instead of VGG

This repository is a personal project in which I reproduce [Multiresolution Knowledge Distillation for Anomaly Detection](https://arxiv.org/pdf/2011.11108.pdf) and exchange the VGG backbone to a ViT backbone. The core idea of the original paper is to distill the knowledge of a pretrained VGG network that scores almost perfectly on the data to a smaller to-be-trained VGG network. This to-be-trained VGG is optimised to detect anomalies. In this project I don't focus as much on the loss function as my goal is just to implement a ViT in a research setting. Furthermore the goal of this project is to complete it within a week (40 hours). Because of the time constraint I limit myself to reproducing and extending only on the CIFAR10 dataset. One other major constraint is that I did not have acces to a GPU cluster for this project, so all work had to be done on google colab. 

### Result base model (VGG)
Training the base model took extremely long (2.5 hours) and due to colab I could not run things in the background / in parallel, thus I did only one run/class. The result of that run can be found in vgg_kdad.ipynb ,  but in summary I trained for class 3, 200 epochs, 1e-3 learning rate, batch size 64. I got an RocAUC of 77.03%, which is almost exactly what was reported in the original paper (77.02%), thus it seemed reasonable to not investigate much further. One thing to note is that the original code-base used the same set for validation and testing, so this is essentially the validation RocAUC. 

### Result my model (VIT)
TBA


### Daily Log

**Monday (5 hours):** Started reading the paper and tried running the code. Because of colab I could not adhere to the package versions so I had to do some bug fixing for that. End of the day I managed to run the code so that was pretty smooth.

**Tuesday (8 hours):** Eventhough I could run the code the night before, the checkpoints were not yet working as the paths were hard-coded. Therefor I decided to restructure the code-base and I fixed the paths. I also started reading into ViTs as I had never really done so before. Also looked for a pretrained ViT on CIFAR10, which I found on Huggingface, and tried to run it.

**Wednesday (8 hours):** Started making my own load data function, because the pretrained ViT was trained on resized CIFAR10 and because the original code-base used the same set for validation and testing. This was suprisingly hard, also to get it working with the pretrained ViT. In the afternoon I started to implement the to-be-trained ViT in pytorch lightning. 




